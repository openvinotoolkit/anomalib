import { fetchClient } from '@anomalib-studio/api';
import { usePipeline } from '@anomalib-studio/hooks';
import { skipToken, useQuery } from '@tanstack/react-query';

import { isNonEmptyString } from '../../../utils';
import { MediaItem } from '../../types';
import { downloadImageAsFile } from './util';

export const useMediaItemInference = (selectedMediaItem: MediaItem) => {
    const { data: pipeline } = usePipeline();
    const selectedModelId = pipeline?.model?.id;

    return useQuery({
        staleTime: 0,
        queryKey: ['inference', selectedMediaItem.id, selectedModelId],
        queryFn:
            isNonEmptyString(selectedModelId) === false
                ? skipToken
                : async () => {
                      const file = await downloadImageAsFile(selectedMediaItem);

                      const formData = new FormData();
                      formData.append('file', file);

                      if (pipeline?.inference_device) {
                          formData.append('device', pipeline.inference_device);
                      }

                      const response = await fetchClient.POST(`/api/projects/{project_id}/models/{model_id}:predict`, {
                          body: formData,
                          params: { path: { project_id: selectedMediaItem.project_id, model_id: selectedModelId } },
                      });

                      return response.data;
                  },
    });
};
