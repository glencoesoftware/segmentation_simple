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
import logging
import pickle
import csv
import sys

from getpass import getpass

import cv2
import numpy as np
import omero
import omero.clients
import omero.scripts as scripts

from omero.grid import ImageColumn, WellColumn, RoiColumn, DoubleColumn
from omero.model import OriginalFileI, PlateI, PlateAnnotationLinkI, ImageI, \
    FileAnnotationI, RoiI, PointI, WellI
from omero.rtypes import rint, rlong, rdouble, rstring
from omero.util.pixelstypetopython import toNumpy


log = logging.getLogger('gs.segmentation_simple')

NS = 'openmicroscopy.org/omero/bulk_annotations'
#NS = 'openmicroscopy.org/omero/measurement'

IMAGE_QUERY = 'select i from Image as i ' \
              'join fetch i.pixels as p ' \
              'join fetch p.pixelsType ' \
              'join fetch i.wellSamples as ws ' \
              'join fetch ws.well as w ' \
              'join fetch w.plate '

DEFAULT_THRESHOLD = 0.7


def standalone_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server', help='OMERO server name')
    parser.add_argument(
        '-p', '--port', type=int, help='OMERO server port', default=4064
    )
    parser.add_argument('-u', '--username', help='OMERO username')
    parser.add_argument('-k', '--session_key', help='OMERO session key')
    parser.add_argument(
        '-v', '--verbose', action='store_const', dest='logging_level',
        const=logging.DEBUG, default=logging.INFO,
        help='Enable more verbose logging'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true', default=False,
        help='Turn on debugging'
    )
    parser.add_argument(
        '--clear_rois', action='store_true', default=False,
        help='Clear ROIs from the server for the object (object_id)'
    )
    parser.add_argument(
        '--save_rois', action='store_true', default=False,
        help='Save ROIs to the server for the object (object_id)'
    )
    parser.add_argument(
        '--threshold', type=float, default=DEFAULT_THRESHOLD,
        help='Stain channel threshold'
    )
    parser.add_argument(
        '--distribute', action='store_true', default=False,
        help='Distribute!'
    )
    parser.add_argument(
        'object_id',
        help='OMERO object or container to analyse (ex. Image:1)'
    )

    args = parser.parse_args()
    if args.username and args.session_key is None:
        args.password = getpass("OMERO password for '%s': " % args.username)
    elif args.username is None and args.session_key is None:
        parser.error('Username or session key must be provided!')
    logging.basicConfig(level=args.logging_level)

    client = connect(args)
    try:
        analyse(client, args)
    finally:
        client.closeSession()

def script_main():
    dataTypes = [rstring('Plate'), rstring('Well'), rstring('Image')]

    client = scripts.client(
        'Segmentation_Simple.py',
        'Simple segmentation on a given Plate, Well or Image',

        scripts.String('Data_Type', optional=False, grouping='1',
                       description='The data type you want to work with.',
                       values=dataTypes,
                       default='Plate'),

        scripts.List('IDs', optional=False, grouping='2',
                     description='List of object IDs').ofType(rlong(0)),

        scripts.Bool('Clear_Existing_ROIs', optional=False, grouping='3',
                     description='Enable debugging?',
                     default=False),

        scripts.Bool('Save_ROIs', optional=False, grouping='4',
                     description='Enable debugging?',
                     default=True),

        scripts.Bool('Debug', optional=False, grouping='5',
                     description='Enable debugging?',
                     default=False),

        version='0.1',
        authors=['Emil Rozbicki', 'Chris Allan'],
        institutions=['Glencoe Software Inc.'],
        contact='support@glencoesoftware.com',
    )

    try:
        if client.getInput('Debug', unwrap=True):
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        script_params = dict()
        for key in client.getInputKeys():
            if client.getInput(key):
                script_params[key] = client.getInput(key, unwrap=True)
        log.debug('Script parameters: %r' % script_params)

        class Arguments(object):
            clear_rois = script_params['Clear_Existing_ROIs']
            save_rois = script_params['Save_ROIs']
            threshold = DEFAULT_THRESHOLD
            object_id = '%s:%s' % \
                (script_params['Data_Type'], script_params['IDs'][0])
        args = Arguments()
        analyse(client, args)

        client.setOutput(
            'Message',
            rstring('Segmentation of %s successful!' % args.object_id)
        )
    finally:
        client.closeSession()

def connect(args):
    client = omero.client(args.server, args.port)
    if args.username is not None:
        session = client.createSession(args.username, args.password)
    else:
        session = client.joinSession(args.session_key)
    ec = session.getAdminService().getEventContext()
    session_key = ec.sessionUuid
    log.debug('Session key: %s' % session_key)
    args.session_key = session_key
    return client

def get_planes(session, pixels):
    pixels_service = session.createRawPixelsStore()

    planes = list()
    dtype = toNumpy(pixels.pixelsType.value.val)
    try:
        ctx = {'omero.group': '-1'}
        pixels_service.setPixelsId(pixels.id.val, True, ctx)
        for c in xrange(pixels.sizeC.val):
            plane = pixels_service.getPlane(0, c, 0)
            plane = np.fromstring(plane, dtype=dtype)
            plane = np.reshape(plane, (pixels.sizeY.val, -1))
            if sys.byteorder == 'little':
                # Data coming from OMERO is always big endian
                plane = plane.byteswap()
            planes.append(plane)
    finally:
        pixels_service.close()
    return planes

def to_rois(cx_column, cy_column, pixels):
    unloaded_image = ImageI(pixels.image.id, False)
    rois = list()
    for index in range(len(cx_column.values)):
        cx = rdouble(cx_column.values[index])
        cy = rdouble(cy_column.values[index])
        roi = RoiI()
        shape = PointI()
        shape.theZ = rint(0)
        shape.theT = rint(0)
        shape.cx = cx
        shape.cy = cy
        roi.addShape(shape)
        roi.image = unloaded_image
        rois.append(roi)
    return rois

def clear_rois(client, pixels):
    session = client.getSession()
    query_service = session.getQueryService()

    ctx = {'omero.group': '-1'}
    params = omero.sys.ParametersI()
    params.addId(pixels.image.id.val)
    rois = query_service.findAllByQuery(
        'select roi from Roi as roi '
        'where roi.image.id = :id',
        params, ctx
    )
    if len(rois) == 0:
        return

    do_all = omero.cmd.DoAll()
    opts = None
    do_all.requests = [omero.cmd.Delete('/Roi', v.id.val, opts) for v in rois]
    handle = session.submit(do_all, ctx)
    try:
        loops = 10
        ms = 1000
        callback = omero.callbacks.CmdCallbackI(client, handle)
        callback.loop(loops, ms)
        response = callback.getResponse()
        if isinstance(response, omero.cmd.ERR):
            raise Exception(response)
        log.info('Clearing of %d ROIs successful!' % len(rois))
    finally:
        handle.close()

def get_columns():
    columns = [
        ImageColumn('Image', '', list()),
        RoiColumn('ROI', '', list()),
        WellColumn('Well', '', list()),
    ]
    column_name = [
        'cellNumber', 'CX', 'CY', 'ellipseCX', 'ellipseCY',
        'polygonArea', 'ellipseArea', 'ellipseR1', 'ellipseR2',
        'excentricity', 'orientation', 'meanCh1', 'meanCh2',
        'totalIntCh1', 'totalIntCh2', 'minValCh1', 'minCX', 'minCY',
        'maxValCh1', 'maxCX', 'maxCY', 'minValCh2', 'minCX2', 'minCY2',
        'maxValCh2', 'maxCX2', 'maxCY2']
    for name in column_name:
        columns.append(DoubleColumn(name, '', list()))
    return columns

def create_file_annotation():
    '''
    Creates a file annotation to represent a set of columns from our
    measurment.
    '''
    file_annotation = FileAnnotationI()
    file_annotation.ns = rstring(NS)
    return file_annotation

def get_table(client, plate_id):
    '''Retrieves the OMERO.tables instance backing our results.'''
    session = client.getSession()
    query_service = session.getQueryService()
    sr = session.sharedResources()
    ctx = {'omero.group': '-1'}

    params = omero.sys.ParametersI()
    params.addString('ns', NS)
    params.addId(plate_id)
    plate = query_service.findByQuery(
        'select p from Plate as p '
        'join fetch p.annotationLinks as a_link '
        'join fetch a_link.child as a '
        'where a.ns = :ns and p.id = :id ',
        params, ctx
    )
    if plate is not None:
        file_annotation = next(plate.iterateAnnotationLinks()).child
        table_original_file_id = file_annotation.file.id.val
        table = sr.openTable(OriginalFileI(table_original_file_id, False))
        log.info('Using existing table: %d' % table_original_file_id)
        return (table, file_annotation)
    return create_table(client, plate_id)

def create_table(client, plate_id):
    '''Create a new OMERO table to store our measurement results.'''
    session = client.getSession()
    update_service = session.getUpdateService()
    sr = session.sharedResources()

    name = '/segmentation_simple.r5'
    table = sr.newTable(1, name)
    if table is None:
        raise Exception(
            'Unable to create table: %s' % name)

    # Retrieve the original file corresponding to the table for the
    # measurement, link it to the file annotation representing the
    # umbrella measurement run, link the annotation to the plate from
    # which it belongs and save the file annotation.
    try:
        table_original_file = table.getOriginalFile()
        table_original_file_id = table_original_file.id.val
        log.info('Created new table: %d' % table_original_file_id)
        unloaded_o_file = OriginalFileI(table_original_file_id, False)
        file_annotation = create_file_annotation()
        file_annotation.file = unloaded_o_file
        unloaded_plate = PlateI(plate_id, False)
        plate_annotation_link = PlateAnnotationLinkI()
        plate_annotation_link.parent = unloaded_plate
        plate_annotation_link.child = file_annotation
        plate_annotation_link = \
            update_service.saveAndReturnObject(plate_annotation_link)
        file_annotation = plate_annotation_link.child
        table.initialize(get_columns())
    except:
        table.close()
        raise
    return (table, file_annotation)

def get_images_by_well(client, image_id):
    ctx = {'omero.group': '-1'}
    session = client.getSession()
    query_service = session.getQueryService()

    params = omero.sys.ParametersI()
    params.addId(image_id)
    images = query_service.findAllByQuery(
        IMAGE_QUERY + 'where w.id = :id', params, ctx)
    return images

def get_image(client, image_id):
    ctx = {'omero.group': '-1'}
    session = client.getSession()
    query_service = session.getQueryService()

    params = omero.sys.ParametersI()
    params.addId(image_id)
    image = query_service.findByQuery(
        IMAGE_QUERY + 'where i.id = :id', params, ctx)
    return image

def unit_of_work(args, image_id):
    args = pickle.loads(args)
    client = omero.client(args.server, args.port)
    client.joinSession(args.session_key)
    try:
        image = get_image(client, image_id)
        plate = next(image.iterateWellSamples()).well.plate
        table, file_annotation = get_table(client, plate.id.val)
        try:
            analyse_image(client, args, table, image)
        except:
            log.error(
                'Error while analysing Image:%d' % image.id.val,
                exc_info=True
            )
        finally:
            table.close()
    except:
        log.error(
            'Error while preparing to analyse Image:%d' % image.id.val,
            exc_info=True
        )
    finally:
        client.closeSession()

def analyse(client, args):
    session = client.getSession()
    query_service = session.getQueryService()
    ctx = {'omero.group': '-1'}

    omero_type, omero_id = args.object_id.split(':')
    omero_object = query_service.get(omero_type, long(omero_id), ctx)

    images = list()
    if isinstance(omero_object, WellI):
        images = get_images_by_well(client, omero_id)
    if isinstance(omero_object, ImageI):
        images.append(get_image(client, omero_id))
    plate = next(images[0].iterateWellSamples()).well.plate

    table, file_annotation = get_table(client, plate.id.val)
    try:
        if args.distribute:
            import omero.work
            dist_args = [
                [pickle.dumps(args), image.id.val] for image in images
            ]
            omero.work.distribute_func(
                'Segmentation_Simple.unit_of_work',
                dist_args
            )
            return file_annotation

        for image in images:
            analyse_image(client, args, table, image)
    finally:
        table.close()
    return file_annotation

def analyse_image(client, args, table, image):
    log.info('Analysing Image:%d' % image.id.val)
    pixels = image.getPrimaryPixels()
    if args.clear_rois:
        clear_rois(client, pixels)
    analyse_planes(client, args, table, image)

def analyse_planes(client, args, table, image):
    session = client.getSession()
    update_service = session.getUpdateService()
    columns = table.getHeaders()
    columns_by_name = dict([(v.name, v) for v in columns])
    pixels = image.getPrimaryPixels()

    pi = 3.14159265359
    img2ThreshVal = args.threshold
    # Read-in images
    img1, img2 = get_planes(session, pixels)

    # Convert to 8-bit
    img1f = img1.astype(float)
    img1f -= (img1.min())
    scale = 255.0 / (img1.max() - img1.min())
    img1_8bit = (scale * img1f).astype(np.uint8)
    img2f = img2.astype(float)
    img2f -= (img1.min() + img2ThreshVal * img2.mean())

    img1Thresh = np.zeros(img1.shape, 'uint8')
    # threshold images
    cv2.threshold(
        img1_8bit, 2, 4, cv2.THRESH_BINARY + cv2.THRESH_OTSU, img1Thresh)
    # find contours
    contoursCh1, hierarchyCh1 = cv2.findContours(
        img1Thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Indenified contour counter
    cellCounter = 0
    # Do the JOB!!!!
    for cnt in contoursCh1:
        area = cv2.contourArea(cnt)  # get area
        if area < 500:
            # remove all tiny contours
            continue
        if area > 3500:
            # and huge contours
            continue
        #fit an ellipse: 1. to discribe the cell, 2. to filter out clusters
        ellipse = cv2.fitEllipse(cnt)
        r1, r2 = ellipse[1]
        ellipseArea = pi * 0.5 * r1 * 0.5 * r2

        if ellipseArea > area + 0.1*area:
            # ellipse fitted to the cluster usually has higher area then
            # cluster itself
            continue
        # define mask for mean intensity computation
        mask = np.zeros(img1_8bit.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        if r1 > r2:
            excentricity = r1 / r2
        else:
            excentricity = r2 / r1
        if excentricity > 1.7:
            continue
        moments = cv2.moments(cnt)
        cellCounter += 1
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        ellipseCx, ellipseCy = ellipse[0]
        meanVal = cv2.mean(img1, mask=mask)
        meanVal2 = cv2.mean(img2, mask=mask)
        minVal, maxVal, min_loc, max_loc = cv2.minMaxLoc(img1, mask=mask)
        maxCx, maxCy = max_loc
        minCx, minCy = min_loc
        minVal2, maxVal2, min_loc2, max_loc2 = cv2.minMaxLoc(img2, mask=mask)
        maxCx2, maxCy2 = max_loc2
        minCx2, minCy2 = min_loc2
        arrayRow = [
            cellCounter, cx, cy, ellipseCx, ellipseCy, moments['m00'],
            ellipseArea, r1, r2, excentricity, ellipse[2],
            meanVal[0], meanVal2[0], meanVal[0] * area, meanVal2[0] * area,
            minVal, minCx, minCy, maxVal, maxCx, maxCy, minVal2,
            minCx2, minCy2, maxVal2, maxCx2, maxCy2
        ]
        # Set Image column
        columns_by_name['Image'].values.append(pixels.image.id.val)
        # Set Well column
        columns_by_name['Well'].values.append(
            next(image.iterateWellSamples()).well.id.val
        )
        for index, value in enumerate(arrayRow):
            columns[index + 3].values.append(float(value))

    log.info('Found %d cells!' % cellCounter)
    ctx = {'omero.group': str(pixels.details.group.id.val)}
    if args.save_rois:
        roi_ids = update_service.saveAndReturnIds(
            to_rois(
                columns_by_name['CX'], columns_by_name['CY'], pixels
            ), ctx
        )
        for roi_id in roi_ids:
            # Set ROI column for each cell
            columns_by_name['ROI'].values.append(roi_id)

    # Write new column data to OMERO.tables
    table.addData(columns)

    with open('output.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow([v.name for v in columns])
        for index in range(len(columns[0].values)):
            writer.writerow([v.values[index] for v in columns])

if __name__ == '__main__':
    if sys.argv[0] == './script':
        script_main()
    else:
        standalone_main()
