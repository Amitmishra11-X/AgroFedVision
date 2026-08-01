import numpy as np


class UAVStatistics:

    def __init__(self):

        self.images = []

        self.ndvi = []

        self.ndre = []

        self.gndvi = []

        self.best_image = None

        self.best_ndvi = -1

    # ------------------------------------------
    # Add one UAV image
    # ------------------------------------------

    def add_image(
        self,
        image_name,
        ndvi_mean,
        ndre_mean=None,
        gndvi_mean=None
    ):

        self.images.append(image_name)

        self.ndvi.append(ndvi_mean)

        if ndre_mean is not None:
            self.ndre.append(ndre_mean)

        if gndvi_mean is not None:
            self.gndvi.append(gndvi_mean)

        if ndvi_mean > self.best_ndvi:

            self.best_ndvi = ndvi_mean

            self.best_image = image_name

    # ------------------------------------------
    # Summary
    # ------------------------------------------

    def summary(self):

        result = {

            "total_images": len(self.images),

            "best_image": self.best_image,

            "ndvi_mean": float(np.mean(self.ndvi))
            if self.ndvi else None,

            "ndvi_std": float(np.std(self.ndvi))
            if self.ndvi else None,

            "ndvi_min": float(np.min(self.ndvi))
            if self.ndvi else None,

            "ndvi_max": float(np.max(self.ndvi))
            if self.ndvi else None,

            "per_image_ndvi": self.ndvi

        }

        if self.ndre:

            result.update({

                "ndre_mean": float(np.mean(self.ndre)),

                "ndre_std": float(np.std(self.ndre))

            })

        if self.gndvi:

            result.update({

                "gndvi_mean": float(np.mean(self.gndvi)),

                "gndvi_std": float(np.std(self.gndvi))

            })

        return result