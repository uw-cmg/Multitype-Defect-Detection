def output_gt_bbox(bboxes,labels,fname):
    """Output the bounding box and class information in
    list of records :   [ label, x1,y1,x2,y2 ]

    Output output file information when everything finishs

    Args:
        bboxes: (list of 4 numbers) [y1,x1,y2,x2]
        label: list of labels
        fname: file name prefix of the output file
    Returns:
        None
    """
    import time
    outputFileName = fname + "_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
    with open(outputFileName, 'w') as outfile:
        for i in range(0, len(bboxes)):
            outfile.write("%s,%.3f,%.3f,%.3f,%.3f \n" %(labels[i],bboxes[i][1],bboxes[i][0],bboxes[i][3],bboxes[i][2]))