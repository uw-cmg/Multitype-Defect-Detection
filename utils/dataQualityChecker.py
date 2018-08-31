"""

This program checks the labelling data quality of human labellings

"""
import sys
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def readInTXTData( TXTPath ):
    """
    read in the labelling data

    input :
    TXTPath : the path of labelling data folder
    :return: the dict of pandas data frame containing all data
    """
    txtDFDict = dict()
    for fileName in os.listdir(TXTPath):
        #print(fileName.strip('.txt'))
        df = pd.read_table(TXTPath + fileName, delim_whitespace=True,
                           names=('class', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'labeltext'))
        newdf = df.reindex(columns=['class', 'Ymin', 'Xmin', 'Ymax', 'Xmax', 'labeltext'])
        txtDFDict[fileName.strip('.txt')] = newdf
    return txtDFDict

def readInCSVData( CSVPath ):
    """
    read in the CSV labelling data

    :input:
    CSVPath : the path of labelling CSV file
    :return: the dict of pandas data frame containing all data
    """
    csvDFDict = dict()
    for fileName in os.listdir(CSVPath):
        #print(fileName.strip('.csv'))
        df = pd.read_csv(CSVPath + fileName)
        csvDFDict[fileName.strip('.csv')] = df
    return csvDFDict

def defectsNumberchecker(txtDFDict, csvDFDict):
    """
    check the length of labelling

    :param txtDFDict: the dataframe dict of all the txtFile Information
    :param csvDFDict: the dataframe dict of all the csvFile Information
    :return: using `assert` so if no output then it means everything is OK
            otherwise it will output the file name of the mismatched file
    """
    #  check the length of each file
    for key in txtDFDict.keys():
        #print(key)
        #print(len(txtDFDict[key]))
        #print(len(csvDFDict[key+"_result"]))
        #print(len(txtDFDict[key]) == (len(csvDFDict[key+"_result"])))
        #print("========================")
        assert len(txtDFDict[key]) == (len(csvDFDict[key+"_result"]))

def defectsRatioDistribution(txtDFDict, csvDFDict):
    """
    Output defectsRatioDistribution Plot

    :param txtDFDict: the dataframe dict of all the txtFile Information
    :param csvDFDict: the dataframe dict of all the csvFile Information
    :return: None
    """
    for key in txtDFDict.keys():
        ratioDF = txtDFDict[key]


if __name__ == '__main__':
    # read in TXT Data
    TXTPATH = "../data/10 images/bounding box/"
    txtDFDict = readInTXTData(TXTPATH)

    # read in CSV Data
    CSVPATH = "../data/10 images/results/"
    csvDFDict = readInCSVData(CSVPATH)
    #print(csvDFDict["200kV_500kx_p2nm_8cmCL_grain1_0068 - Copy_result"])

    # check the length of each labelling matched
    defectsNumberchecker(txtDFDict, csvDFDict)

    # Plot the distribution of Major axis and Minor axis
    ratioDFDict = dict()
    for key in txtDFDict.keys():
        #ratioDF = csvDFDict[key+"_result"]["Major"] / csvDFDict[key+"_result"]["Minor"]
        ratioDF = pd.concat([txtDFDict[key]['labeltext'], csvDFDict[key+"_result"]["Major"] / csvDFDict[key+"_result"]["Minor"]], axis=1, keys=['labeltext', 'ratios'])
        ratioDFDict[key] = ratioDF
        #print(key)


    # plotting and save figures
    for key in txtDFDict.keys():
        # create color list
        colorList = list()
        for index, row  in ratioDFDict[key].iterrows():
            if row['labeltext'] == '111':
                colorList.append('red')
            elif row['labeltext'] == '100':
                colorList.append('green')
            else:
                colorList.append('blue')

        #print(colorList)
        fignow = plt.figure()
        plt.scatter(ratioDFDict[key].index.tolist(),ratioDFDict[key]['ratios'],color=colorList)
        plt.title(key)
        plt.xlabel("defect number index")
        plt.ylabel("ratios of major and minor axis")
        #plt.legend(('111','100','bd'),loc='upper left')
        L = plt.legend(loc='upper left')
        L.get_texts()[0].set_text('\n read for 111 \n green for 100 \n blue for bd')
        #plt.text(1,3,'read for 111 \n green for 100 \n blue for bd', fontsize=11, color='black')
        #plt.show()
        fignow.savefig(key + '.png')


    # Area

    # plotting and save figures
    for key in txtDFDict.keys():
        # create color list
        colorList = list()
        for index, row  in ratioDFDict[key].iterrows():
            if row['labeltext'] == '111':
                colorList.append('red')
            elif row['labeltext'] == '100':
                colorList.append('green')
            else:
                colorList.append('blue')

        #print(colorList)
        fignow = plt.figure()
        plt.scatter(ratioDFDict[key].index.tolist(),csvDFDict[key+"_result"]["Area"],color=colorList)
        plt.title(key)
        plt.xlabel("defect number index")
        plt.ylabel("Area of defects in pixels")
        #plt.legend(('111','100','bd'),loc='upper left')
        L = plt.legend(loc='upper left')
        L.get_texts()[0].set_text('\n read for 111 \n green for 100 \n blue for bd')
        #plt.text(1,3,'read for 111 \n green for 100 \n blue for bd', fontsize=11, color='black')
        #plt.show()
        fignow.savefig('Area_' + key  + '.png')




