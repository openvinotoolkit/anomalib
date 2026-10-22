// Python owns parsing and aggregation; this file only presents prepared data.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".benchmark-history").forEach((panel) => {
    const data = JSON.parse(
      panel.querySelector(".benchmark-chart-data").textContent,
    );
    const chart = panel.querySelector(".benchmark-chart");
    const metricSelect = panel.querySelector(".benchmark-metric");
    const categorySelect = panel.querySelector(".benchmark-category");
    const versionSelect = panel.querySelector(".benchmark-version");
    const tbody = panel.querySelector(".benchmark-category-table tbody");
    const summary = document.createElement("p");
    summary.className = "benchmark-chart-summary";
    summary.setAttribute("aria-live", "polite");
    chart.before(summary);

    function updateTable() {
      tbody.replaceChildren();
      for (const row of data.rows[versionSelect.value]) {
        const tr = tbody.insertRow();
        row.forEach((value, index) => {
          const cell = tr.insertCell();
          if (index > 0) {
            cell.textContent = value.toFixed(3);
            return;
          }
          const button = document.createElement("button");
          button.type = "button";
          button.className = "benchmark-category-link";
          button.textContent = value;
          button.addEventListener("click", () => {
            categorySelect.value = String(
              data.series.findIndex((series) => series.label === value),
            );
            render();
          });
          cell.append(button);
        });
      }
    }

    function render() {
      const metric = data.metrics.find(
        (item) => item.key === metricSelect.value,
      );
      const selected =
        categorySelect.value === "all"
          ? data.series
          : [data.series[Number(categorySelect.value)]];
      const styles = getComputedStyle(document.documentElement);
      const color = (name) => styles.getPropertyValue(name).trim();
      const latest = selected[0].values[metric.key].at(-1);
      summary.textContent =
        categorySelect.value === "all"
          ? `${metric.label} · Category average and ${data.series.length - 1} categories`
          : `${selected[0].label} · Latest ${latest === null ? "not available" : latest.toFixed(3)} · ${data.versions.length} runs / versions`;
      Plotly.react(
        chart,
        selected.map((series) => ({
          x: data.versions,
          y: series.values[metric.key],
          name: series.label,
          customdata: data.category_counts,
          type: "scatter",
          mode: "lines+markers",
          connectgaps: false,
          line: {
            width: series === data.series[0] ? 3 : 1.5,
            ...(series === data.series[0]
              ? { color: color("--pst-color-primary") }
              : {}),
          },
          marker: { size: 6 },
          hovertemplate:
            "%{x}<br>" +
            metric.label +
            ": %{y:.4f}<br>Run coverage: %{customdata} categories<extra>%{fullData.name}</extra>",
        })),
        {
          margin: { t: 24, r: 24, b: 64, l: 64 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: {
            family: getComputedStyle(panel).fontFamily,
            size: 13,
            color: color("--pst-color-text-base"),
          },
          xaxis: {
            title: { text: "Run / anomalib version" },
            type: "category",
            showgrid: false,
            tickangle: 0,
            nticks: 6,
            automargin: true,
          },
          yaxis: {
            title: { text: "Score" },
            tickformat: ".3f",
            gridcolor: color("--pst-color-border"),
            zeroline: false,
            automargin: true,
            autorangeoptions: { clipmin: 0, clipmax: 1 },
          },
          showlegend: selected.length > 1,
          legend: { orientation: "h" },
          hovermode: "closest",
        },
        {
          responsive: true,
          displaylogo: false,
          scrollZoom: false,
          modeBarButtonsToRemove: ["select2d", "lasso2d"],
        },
      );
    }

    metricSelect.addEventListener("change", render);
    categorySelect.addEventListener("change", render);
    versionSelect.addEventListener("change", updateTable);
    new MutationObserver(render).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    updateTable();
    render();
    chart.on("plotly_click", (event) => {
      versionSelect.value = event.points[0].x;
      updateTable();
      panel.querySelector(".benchmark-categories").open = true;
    });
  });

  function openModel(hash = location.hash) {
    if (!hash) return;
    const panel = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (panel?.classList.contains("benchmark-history")) {
      panel.querySelector(".benchmark-categories").open = true;
      panel.focus({ preventScroll: true });
    }
  }
  document.querySelectorAll('a[href^="#benchmark-"]').forEach((link) => {
    link.addEventListener("click", () => openModel(link.hash));
  });
  window.addEventListener("hashchange", () => openModel());
  openModel();
});
