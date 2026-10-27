let locationWatcher = null;

function startLocationTracking() {

    if (!navigator.geolocation) {
        updateLocationStatus("GPS not supported");
        return;
    }

    locationWatcher = navigator.geolocation.watchPosition(

        function(position) {

            const latitude =
                position.coords.latitude;

            const longitude =
                position.coords.longitude;

            const accuracy =
                position.coords.accuracy;

            const speed =
                position.coords.speed;

            updateLocationUI(
                latitude,
                longitude,
                accuracy,
                speed
            );

        },

        function(error) {

            console.error("GPS error:", error);

            updateLocationStatus(
                "Location permission required"
            );

        },

        {
            enableHighAccuracy: true,
            maximumAge: 5000,
            timeout: 10000
        }
    );
}


function stopLocationTracking() {

    if (locationWatcher !== null) {

        navigator.geolocation.clearWatch(
            locationWatcher
        );

        locationWatcher = null;
    }

    updateLocationStatus("GPS stopped");
}


function updateLocationUI(
    latitude,
    longitude,
    accuracy,
    speed
) {

    document.getElementById("latitude").textContent =
        latitude.toFixed(6);

    document.getElementById("longitude").textContent =
        longitude.toFixed(6);

    document.getElementById("accuracy").textContent =
        Math.round(accuracy) + " m";

    if (speed !== null) {

        const speedKmh =
            speed * 3.6;

        document.getElementById("speed").textContent =
            speedKmh.toFixed(1) + " km/h";
    }

    updateLocationStatus("GPS active");
}


function updateLocationStatus(message) {

    const element =
        document.getElementById("gpsStatus");

    if (element) {
        element.textContent = message;
    }
}