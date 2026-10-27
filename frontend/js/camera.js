let cameraStream = null;

async function startCamera() {
    const video = document.getElementById("cameraPreview");

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showCameraStatus("Camera API is not supported.");
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: {
                    ideal: "environment"
                },
                width: {
                    ideal: 1280
                },
                height: {
                    ideal: 720
                }
            },
            audio: false
        });

        video.srcObject = cameraStream;

        showCameraStatus("Camera active");

        document.getElementById("cameraIndicator").className =
            "status-dot online";

    } catch (error) {
        console.error("Camera error:", error);

        showCameraStatus(
            "Camera permission denied or camera unavailable."
        );
    }
}


function stopCamera() {
    if (!cameraStream) return;

    cameraStream.getTracks().forEach(track => {
        track.stop();
    });

    cameraStream = null;

    const video = document.getElementById("cameraPreview");

    if (video) {
        video.srcObject = null;
    }

    showCameraStatus("Camera stopped");
}


function showCameraStatus(message) {
    const element = document.getElementById("cameraStatus");

    if (element) {
        element.textContent = message;
    }
}