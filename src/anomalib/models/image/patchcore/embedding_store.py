# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Write-through embedding staging for PatchCore.

PatchCore collects the patch embeddings of every training batch before the
memory bank is subsampled at the end of training. Historically this staging
area was a plain list of per-batch tensors, which meant the full embedding
matrix had to be materialized a second time by ``torch.vstack`` when training
ended - doubling peak memory at exactly the moment the bank is largest.

This module provides :class:`EmbeddingStore`, a staging area that copies each
batch into a single tensor as it arrives so that no second copy is ever
needed. See gh-3815 for the original report.

Example:
    >>> import torch
    >>> store = EmbeddingStore()
    >>> store.push(torch.randn(8, 512))
    >>> store.push(torch.randn(4, 512))
    >>> store.consolidate().shape
    torch.Size([12, 512])
"""

import torch


class EmbeddingStore:
    """Staging buffer that collects training embeddings without doubling memory.

    The store has two modes:

    - **List mode** (the fallback, and the initial state): incoming batches are
      appended to a plain Python list, exactly like the historical
      implementation. Consolidation then stacks them once with
      ``torch.vstack``. This is the behavior of the store until an exact
      capacity is reserved with :meth:`reserve`.
    - **Staging mode** (the optimization): once the total number of embedding
      rows is known - from the trainer, after the first batch reveals how many
      rows each image contributes - the store allocates one tensor of exactly
      that capacity and copies every subsequent batch into it. Consolidation
      hands the staging tensor over without any copy when it is exactly
      filled.

    Because the staging tensor is reserved only when its capacity is known
    exactly, the store never performs growth copies in staging mode in the
    normal engine-driven training flow. If more rows arrive than were reserved
      - a dataloader that lies about its length, for example - the store
      falls back to list mode for the overflow, which is always correct.

    Row order is identical to ``torch.vstack`` over the pushed batches in
    every mode, so the resulting memory bank is byte-identical to the
    historical implementation.
    """

    def __init__(self) -> None:
        self._chunks: list[torch.Tensor] = []
        self._staging: torch.Tensor | None = None
        self._rows = 0
        self._reserved_rows = 0
        self._overflowed = False
        self._staged_batches_count = 0

    @property
    def num_rows(self) -> int:
        """int: Number of embedding rows pushed so far."""
        return self._rows

    @property
    def num_batches(self) -> int:
        """int: Number of batches pushed so far."""
        return len(self._chunks) + self._staged_batches_count

    def __len__(self) -> int:
        """Return the number of batches pushed so far.

        Matches the historical ``list`` interface, where ``len`` counted
        batches (not rows).
        """
        return self.num_batches

    def reserve(self, num_rows: int, num_features: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate the staging tensor with exactly ``num_rows`` rows.

        After a successful reservation the store is in staging mode, and
        batches are copied into the staging tensor as they arrive. Reservation
        is only allowed before any batch has been pushed.

        Args:
            num_rows (int): Total number of embedding rows the run will push.
                A non-positive value leaves the store in list mode.
            num_features (int): Width of each embedding row.
            device (torch.device): Device to allocate on.
            dtype (torch.dtype): Dtype of the embeddings.

        Raises:
            RuntimeError: If any batch has already been pushed, or if the store
                already holds a different reservation.
            ValueError: If ``num_rows`` is negative.
        """
        if num_rows < 0:
            msg = f"Cannot reserve a negative number of rows ({num_rows})."
            raise ValueError(msg)
        if self._chunks:
            msg = "Cannot reserve embedding storage after batches have been pushed."
            raise RuntimeError(msg)
        if self._staging is not None:
            if self._reserved_rows == num_rows and self._staging.shape[1] == num_features:
                return  # idempotent re-reservation of the same capacity
            msg = (
                "Cannot reserve embedding storage with a different capacity "
                f"(reserved rows={self._reserved_rows}, requested rows={num_rows})."
            )
            raise RuntimeError(msg)
        if num_rows == 0:
            return  # stay in list mode
        self._staging = torch.empty((num_rows, num_features), device=device, dtype=dtype)
        self._reserved_rows = num_rows

    @property
    def is_list_mode(self) -> bool:
        """bool: Whether the store still behaves like the historical list."""
        return self._staging is None and not self._overflowed

    @property
    def first_chunk(self) -> torch.Tensor | None:
        """torch.Tensor | None: First pushed chunk, if the store is in list mode."""
        if self._staging is not None or self._overflowed or not self._chunks:
            return None
        return self._chunks[0]

    @property
    def is_staged(self) -> bool:
        """bool: Whether the store is in staging mode (exact reservation)."""
        return self._staging is not None and not self._overflowed

    def upgrade(self, num_rows: int, num_features: int, device: torch.device, dtype: torch.dtype) -> bool:
        """Move already-pushed chunks into an exact-size staging tensor.

        Called once the total row count of the run is known - typically after
        the first training batch reveals how many rows each image contributes.
        The chunks pushed so far (usually a single small batch) are copied into
        the new staging tensor and released, after which every further batch is
        written through.

        Args:
            num_rows (int): Total number of embedding rows the run will push.
            num_features (int): Width of each embedding row.
            device (torch.device): Device to allocate on.
            dtype (torch.dtype): Dtype of the embeddings.

        Returns:
            bool: ``True`` if the store is now in staging mode, ``False`` if
            the upgrade was refused (incompatible or impossible request) and
            the store stays in list mode.
        """
        if self._staging is not None or self._overflowed:
            return False
        if not self._chunks:
            self.reserve(num_rows=num_rows, num_features=num_features, device=device, dtype=dtype)
            return self.is_staged
        first = self._chunks[0]
        if num_rows < self._rows or num_features != first.shape[1] or first.device != device or first.dtype != dtype:
            return False
        staging = torch.empty((num_rows, num_features), device=device, dtype=dtype)
        moved_batches = len(self._chunks)
        offset = 0
        for chunk in self._chunks:
            staging[offset : offset + chunk.shape[0]].copy_(chunk, non_blocking=False)
            offset += chunk.shape[0]
        self._chunks = []
        self._staging = staging
        self._reserved_rows = num_rows
        self._staged_batches_count += moved_batches
        return True

    def push(self, embedding: torch.Tensor) -> None:
        """Stage one batch of embedding rows.

        In list mode the batch is appended to a list (historical behavior).
        In staging mode it is copied into the staging tensor at the current
        write offset. If the reserved capacity is exceeded, the store falls
        back to list mode for good: everything staged so far is moved to the
        list and all further batches are appended there.

        Args:
            embedding (torch.Tensor): 2D tensor of embedding rows
                ``(rows, num_features)``.

        Raises:
            ValueError: If the batch is not 2D.
            RuntimeError: If a batch arrives with a different width or dtype
                than the staging tensor.
        """
        if embedding.ndim != 2:
            msg = f"Embeddings must be 2D, got shape {tuple(embedding.shape)}."
            raise ValueError(msg)
        if self._staging is None or self._overflowed:
            self._chunks.append(embedding)
            self._rows += embedding.shape[0]
            return
        if embedding.shape[1] != self._staging.shape[1]:
            msg = (
                f"Embedding width changed mid-training: staging has {self._staging.shape[1]} features, "
                f"batch has {embedding.shape[1]}."
            )
            raise RuntimeError(msg)
        if embedding.dtype != self._staging.dtype:
            msg = f"Embedding dtype changed mid-training: staging is {self._staging.dtype}, batch is {embedding.dtype}."
            raise RuntimeError(msg)
        start, end = self._rows, self._rows + embedding.shape[0]
        if end > self._reserved_rows:
            # Dataloader delivered more rows than reserved. Fall back to list
            # mode so correctness is never at risk; the staging tensor is
            # compacted into the chunk list and released.
            self._overflowed = True
            staged_rows = self._rows
            staged = self._staging[:staged_rows]
            if staged_rows > 0:
                # Clone so the oversized reservation is actually released when
                # the staging tensor is dropped; the view would pin its storage.
                self._chunks.append(staged.clone())
            self._staging = None
            self._reserved_rows = 0
            self._chunks.append(embedding)
            self._rows += embedding.shape[0]
            return
        self._staging[start:end].copy_(embedding, non_blocking=False)
        self._rows += embedding.shape[0]
        self._staged_batches_count += 1

    def consolidate(self) -> torch.Tensor:
        """Return all staged rows as one tensor in ``torch.vstack`` order.

        In staging mode with an exact fill this hands over the staging tensor
        itself - no copy, no second allocation. In list mode this stacks the
        chunks once, exactly like the historical implementation.

        Returns:
            torch.Tensor: 2D tensor of all pushed rows.

        Raises:
            ValueError: If nothing has been pushed.
        """
        if self._rows == 0:
            msg = "No embeddings collected. Run model in training mode first."
            raise ValueError(msg)
        if self._staging is not None and not self._overflowed:
            # Exact fill hands over the staging tensor itself; a short fill
            # (e.g. ``drop_last`` skipped the tail batch) hands over a view.
            # Neither allocates a second full-size copy; the coreset sampler
            # that runs next reads the tensor and replaces the memory bank with
            # a much smaller selection, which releases this storage.
            return self._staging[: self._rows]
        return torch.vstack(self._chunks)

    def clear(self) -> None:
        """Reset the store so it can stage a fresh training run."""
        self._chunks = []
        self._staging = None
        self._rows = 0
        self._reserved_rows = 0
        self._overflowed = False
        self._staged_batches_count = 0
