######### LOAD AND PREPROCESS

def preprocessImage(in_image, visualize=False):
    """
    Preprocess image for the purpose of text bubble detection
    
    PARAMETERS
    ----------
    in_image : numpy.ndarray with shape (l, w, c)
    visualize : display original and preprocessed image
    
    RETURN
    ------
    out_image : numpy.ndarray with shape (l, w, c)
    
    """
    
    out_image = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    out_image = cv2.adaptiveThreshold(out_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10)
    out_image = cv2.erode(out_image, np.ones((3,3)), iterations = 1)

    if visualize:
        fig, axes = plt.subplots(ncols=2, figsize=(15,10))
        axes[0].imshow(in_image)
        axes[0].set_title('Original image')
        axes[1].imshow(out_image, cmap='gray')
        axes[1].set_title('Preprocessed image')
    
    return out_image