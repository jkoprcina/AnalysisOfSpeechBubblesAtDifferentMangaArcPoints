######### FILTER BUBBLES FIRST PASS

def connectedComponents(image, iom_threshold=0.7, offset = 10, visualize=False):
    """
    Rough estimation of bounding boxes using connect_components, 
    contains noisy/redundent information, however it always includes 
    all speech bubbles / text boxes among estimates so we use this 
    as a first pass filter
    
    PARAMETERS
    ----------
    image : numpy.ndarray with shape (l, w, c)
    iom_threshold : float, determines % of overlap necessary to discard overlapping bounding boxes
    offset : int, extends bounding boxes to include edges of panels
    
    RETURN
    ------
    box_candidates : list of tuples (y0, y1, x0, x1) denoting bounding box upper-left and bottom-right corners 
    
    """
    box_stats = cv2.connectedComponentsWithStats(image, 4, cv2.CV_32S)[2]
    box_area = box_stats[:, 4]
    
    l, w = image.shape
    area_condition = (box_area > l*w/20**2) & (box_area < l*w/7**2)
    filtered_stats = box_stats[area_condition]
    
    box_candidates = []  
    for x,y,w,h in filtered_stats[:, :4]:
        box_candidates.append((y-offset, y+h+offset, x-offset, x+w+offset))

    if visualize:
        # Messy grid visualization of box_candidates
        nrows, ncols = 5, int(len(box_candidates)/5) + 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
        
        ix = 0
        for row in axes:
            for col in row: 
                if ix < len(box_candidates):
                    pts = box_candidates[ix]
                    y0,y1,x0,x1 = pts
                    col.imshow(image[y0:y1,x0:x1], cmap='gray')
                col.get_xaxis().set_ticks([])
                col.get_yaxis().set_ticks([])
                ix += 1
        plt.show()
    
    return box_candidates