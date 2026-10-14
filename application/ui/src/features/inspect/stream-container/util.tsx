import { RefObject } from 'react';

export const captureVideoFrame = (videoRef: RefObject<HTMLVideoElement | null>) => {
    const { promise, resolve, reject } = Promise.withResolvers<File>();

    const video = videoRef.current;
    if (!video) {
        reject('Video element not found');
        return promise;
    }

    // Create an in-memory canvas
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
        reject('Failed to get canvas context');
        return promise;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        if (!blob) {
            reject('Failed to capture frame');
            return;
        }

        const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
        resolve(file);
    }, 'image/jpeg');

    return promise;
};
