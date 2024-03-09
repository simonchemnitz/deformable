import numpy as np
import probmetrics as pm
import imstat as imstat
import misc as misc
import skimage.filters as filters
import scipy.ndimage as ndi


class Deformable:
    """
    Deformable-Model for pattern segmentation
    """

    def __init__(self) -> None:
        self.model = None
        self.model_boundary = None
        self.likelihoodmap = None
        self.image = None
        self.optimal_scale = None

    def initial_model(self, model: np.ndarray):
        """
        Initialise with a given model (region)
        """
        self.model = model

    def _optimise_scale(self) -> int:
        """
        Optimize radius for optimal scale
        """
        ### TODO write code here

        opt_scale = None
        self.optimal_scale = opt_scale

        return opt_scale

    def _calculate_likelihoodmap(
        self,
        image: np.ndarray,
        model: np.ndarray,
        s_opt: int,
        sigma: float,
        gabors: list,
        mode="reflect",
        cval="0",
    ) -> np.ndarray:
        """
        Calculate the likelihood map for the current model
        Parameters:
        -----------

        gabors: list
            List of gabor kernels skimage.filters.gabor_kernel
        """
        # Calculate ientesity pdf of the model
        model_pdf = imstat.region_intensity_pdf(
            image, model, sigma
        )  ###TODO update func

        # empty likelihood map
        L_map = np.zeros(shape=image.shape)

        # For each filter update the likelihood map
        # with L_n
        for g_filter in gabors:
            # Gabor likelihood map to update for each intensity
            L_n = np.zeros(shape=image.shape)

            # Convolve the gabor filter with the image
            tilde_image = ndi.convolve(image, np.real(g_filter), mode=mode, cval=cval)

            # For each intensity update the gabor likelihood
            # with L_i
            for i in range(255):
                # Normalisation constant
                norm_c = self._normalization_constant()

                # Calculate exponential map
                P_i = (1 / norm_c) * np.exp(-((i - tilde_image) ** 2) / (2 * sigma**2))

                # Calculate texton probabilities
                PT_i = ndi.convolve(P_i, np.ones(shape=(s_opt, s_opt)))

                # Multiply with model probability for intensity i
                L_i = PT_i * model_pdf[i]
                L_i = np.sqrt(L_i)

                # Update L_n map
                L_n = L_n + L_i

            # Update L_map
            L_map = L_map * L_n

        return L_map

    def _normalization_constant(self):
        ###TODO
        return None

    def _updatemodel(self) -> np.ndarray:
        """
        Update the model
        """
        ### TODO write code here

    def segment(self, image: np.ndarray, model: np.ndarray) -> np.ndarray:
        """
        Segment an image
        """
        self.initial_model(model=model)


if __name__ == "__main__":
    # Load image:
    img = None
    model = None

    # Define initial model
    dm = Deformable()
    dm.segment(img, model)
