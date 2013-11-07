# -*- coding: utf-8 -*-

'''

:author: Emil Rozbicki <emil@glencoesoftware.com>
:author: Chris Allan <callan@glencoesoftware.com>

Simple segmentation using OpenCV and OMERO
Copyright (C) 2013 Glencoe Software, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import argparse
import csv
import numpy as np
import cv2
import os

from getpass import getpass

import omero
import omero.clients

#from matplotlib import pyplot as plt
#from scipy.ndimage import label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', help='OMERO server name')
    parser.add_argument('-p', '--port', type=int, help='OMERO server port')
    parser.add_argument('-u', '--username', help='OMERO username')
    parser.add_argument('-k', '--session_key', help='OMERO session key')
    parser.add_argument(
        'object_id',
        help='OMERO object or container to analyse (ex. Image:1)'
    )
    args = parser.parse_args()
    if args.username and args.session_key is None:
        args.password = getpass("OMERO password for '%s': " % args.username)
    elif args.username is None and args.session_key is None:
        parser.error('Username or session key must be provided!')
    client = connect(args)
    try:
        analyse(client, args)
    finally:
        client.closeSession()

def connect(args):
    client = omero.client(args.server, args.port)
    if args.username is not None:
        session = client.createSession(args.username, args.password)
    else:
        session = client.joinSession(args.session_key)
    print session.getAdminService().getEventContext().sessionUuid
    return client

def analyse(client, args):
    session = client.getSession()
    query_service = session.getQueryService()
    omero_type, omero_id = args.object_id.split(':')
    omero_object = query_service.get(omero_type, omero_id)

    ctx = {'omero.group': '-1'}
    if isinstance(omero_object, omero.model.Image):
        params = omero.sys.ParametersI()
        params.addId(omero_id)
        pixels = query_service.findByQuery(
            'select p from Pixels as p ' \
            'join fetch p.pixelsType '  \
            'where p.image.id = :id',
            params, ctx
        )
    print pixels

    pi = 3.14159265359
    img2ThreshVal = 0.7
    #Read-in images

    dirPath =  '/Users/emilrozbicki/Documents/Data/JCB/team/plate 11001_Plate_136/TimePoint_1'
    fileNameCh1 = 'plate 11001_A01_s1_w1.TIF'
    fileNameCh2 = 'plate 11001_A01_s1_w2.TIF'
    img1 = cv2.imread(os.path.join(dirPath, fileNameCh1), 2)
    img2 = cv2.imread(os.path.join(dirPath, fileNameCh2), 2)
    imageID = fileNameCh1
    #Convert to 8-bit
    img1f = img1.astype(float)
    img1f -= (img1.min())
    img1_8bit = ((255. / (img1.max() - img1.min())) * img1f).astype(np.uint8)
    img2f = img2.astype(float)
    print img2.max(), img2.min(), img2.mean()
    img2MaxVal = img2.max()
    if img2.max() < 400:
        img2MaxVal = 400
    img2f -= (img1.min()+img2ThreshVal*img2.mean())
    img2_8bit = ((255. / (img2MaxVal - (img2.min()+img2ThreshVal*img2.mean()))) * img2f).astype(np.uint8)

    #img1_8bit = np.uint8(img1_temp)
    print img1_8bit.dtype, img1_8bit.max()
    img1Thresh = np.zeros(img1.shape,'uint8')
    #cv2.convertScaleAbs(img1,img1_8bit,0.5,-50)
    #cv2.convertScaleAbs(img2,img2_8bit,0.5,-50)
    #threshold images
    cv2.threshold(img1_8bit, 2,4, cv2.THRESH_BINARY+cv2.THRESH_OTSU, img1Thresh)
    #find contours
    contoursCh1, hierarchyCh1 = cv2.findContours(img1Thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #images to store results
    imgAllContours = cv2.cvtColor(img1_8bit, cv2.COLOR_GRAY2BGR)
    imgResult = cv2.cvtColor(img1_8bit, cv2.COLOR_GRAY2BGR)
    imgResultCh2 = cv2.cvtColor(img2_8bit, cv2.COLOR_GRAY2BGR)
    #For compuation of mean intenisty (Channel1 and Channel2 together)
    colorImg = np.zeros_like(imgAllContours)
    colorImg[:,:,0] = img1_8bit
    colorImg[:,:,1] = img2_8bit
    #Array holding analyis result
    resultArray = [];
    arrayRow = ['imageID', 'cellNumber', 'CX', 'CY', 'ellipseCX', 'ellipseCY', 
        'polygonArea', 'ellipseArea', 'ellipseR1', 'ellipseR2', 'excentricity',
        'orientation', 'meanCh1', 'meanCh2', 'totalIntCh1', 'totalIntCh2',
        'minValCh1', 'minCX', 'minCY', 'maxValCh1', 'maxCX', 'maxCY','minValCh2',
        'minCX2', 'minCY2', 'maxValCh2', 'maxCX2', 'maxCY2']
    resultArray.append(arrayRow)
    #Indenified contour counter
    cellCounter = 0;
    collocalizationCounter = 0;
    #Do the JOB!!!!
    for cnt in contoursCh1:
        area = cv2.contourArea(cnt)  #get area
        if area <500:   #remove all tiny contours
            continue
        if area > 3500: #and huge contours
            continue
        ellipse = cv2.fitEllipse(cnt)       #fit an ellipse: 1. to discribe the cell, 2. to filter out clusters
        cv2.ellipse(imgAllContours, ellipse, (0,255,0),2)
        cv2.drawContours(imgAllContours,cnt,-1,(255,0,0),2)
        r1, r2 = ellipse[1]
        ellipseArea = pi*0.5*r1*0.5*r2
        if ellipseArea > area + 0.1*area:  #ellipse fitted to the cluster usually has higher area then cluster itself
            continue
        #define mask for mean intensity computation    
        mask = np.zeros(img1_8bit.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        if r1 > r2:
            excentricity = r1/r2
        else:
            excentricity = r2/r1
        if excentricity > 1.7:
            continue
        moments = cv2.moments(cnt)
        cellCounter += 1
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])
        ellipseCx, ellipseCy = ellipse[0]
        meanVal = cv2.mean(img1,mask = mask)
        meanVal2 = cv2.mean(img2,mask = mask)
        minVal, maxVal, min_loc, max_loc = cv2.minMaxLoc(img1,mask = mask)
        maxCx, maxCy = max_loc
        minCx, minCy = min_loc
        minVal2, maxVal2, min_loc2, max_loc2 = cv2.minMaxLoc(img2,mask = mask)
        maxCx2, maxCy2 = max_loc2
        minCx2, minCy2 = min_loc2
        arrayRow = [imageID, cellCounter, cx, cy, ellipseCx, ellipseCy, 
        moments['m00'], ellipseArea, r1, r2, excentricity,ellipse[2], meanVal[0],
        meanVal2[0], meanVal[0]*area, meanVal2[0]*area,minVal, minCx, minCy,
        maxVal, maxCx, maxCy,minVal2, minCx2, minCy2, maxVal2, maxCx2, maxCy2]
        resultArray.append(arrayRow)
        #print arrayRow
        if meanVal2[0] > 125:
             cv2.ellipse(imgResult, ellipse, (0,0,255),-1)
             collocalizationCounter += 1
        cv2.ellipse(imgResult, ellipse, (0,255,0),2)
        cv2.drawContours(imgResult,cnt,-1,(255,0,0),2)
        cv2.drawContours(imgResultCh2,cnt,-1,(255,0,0),2)
    
    print
    print 'found %i cells, %i show collocalization' % (cellCounter, collocalizationCounter) 
    plt.subplot(121),plt.imshow(imgResult, 'gray'),plt.title('Ch1, blue - high intenisty in Ch2')
    plt.subplot(122),plt.imshow(imgResultCh2, 'gray'),plt.title('Channel2')
    plt.show()

    with open('output.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(resultArray)

if __name__ == '__main__':
    main()