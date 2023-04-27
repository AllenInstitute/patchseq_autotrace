import glymur
import numpy as np
from scipy.ndimage import zoom
import argschema as ags
from argschema.fields.files import validate_input_path
import cv2
import marshmallow as mm
import time
import tifffile


# Code based on:
# https://github.com/AllenInstitute/jp2_to_tiff

class InputFileRetries(mm.fields.Str):
    def _validate(self, value):
        n_retries = 5
        success = False
        for i in range(n_retries):
            try:
                validate_input_path(value)
                success = True
                break
            except mm.exceptions.ValidationError as e:
                error = e
                time.sleep(2)
                continue
        if not success:
            raise (error)


class ConverterSchema(ags.ArgSchema):
    input_jp2 = InputFileRetries(
        required=True,
        description="path to jp2 input file")
    x_ind = ags.fields.Int(
        required=True,
        description="start index for x"
    )
    y_ind = ags.fields.Int(
        required=True,
        description="start index for y"
    )
    downsample = ags.fields.Int(
        required=False,
        default=1,
        missing=1,
        description="downsample factor per axis")
    tiff_output_file = ags.fields.OutputFile(
        required=True,
        description="destination for output tiff")
    block_read = ags.fields.Int(
        required=False,
        default=2,
        missing=2,
        description="read jp2 in n x n chunks to avoid glymur death")


def indices_from_shape(sz, n):
    sr = np.array_split(np.arange(sz[0]), n)
    sc = np.array_split(np.arange(sz[1]), n)
    indices = []
    for r in sr:
        for c in sc:
            indices.append([r.min(), r.max() + 1, c.min(), c.max() + 1])
    return indices


def load_in_chunks(jp2, indices):
    ims = [jp2[i[0]: i[1], i[2]: i[3]] for i in indices]
    return ims


def concatenate_chunks(ims, n):
    rims = []
    for i in range(n):
        rims.append(np.hstack(ims[i * n: (i + 1) * n]))
    im = np.vstack(rims)
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    #         im = np.concatenate([im,im,im],axis=2)
    return im


def downsample(im, factor):
    """im has 3 channels
    """
    nchan = im.shape[2]
    dsim = []
    for i in range(nchan):
        dsim.append(
            zoom(
                im[:, :, i],
                1.0 / factor,
                order=1))
    dsim = np.array(dsim)
    dsim = np.moveaxis(dsim, 0, 2)
    return dsim


class JP2ToTiff(ags.ArgSchemaParser):
    default_schema = ConverterSchema

    def __init__(self, args):
        """
        convert a jp2 image to a tif image

        :param args: a python dictionary with the following key/values
            :key x_ind: x dimension index for image ROI
            :key y_in: y dimension index for image ROI
            :key input_jp2: path to jp2 to convert to tiff
            :key downsample: size at which to downsample, 1 = no downsampling
            :key block_read: load jp2 in blocks to avoid crashes
            :key tiff_output_file: path where to write output tif file
        """
        self.args = args

    def run(self):
        # open the file
        jp2 = glymur.Jp2k(self.args['input_jp2'])

        # get indices for chunked read
        indices = indices_from_shape(jp2.shape, self.args['block_read'])

        # read in chunks to numpy arrays
        ims = load_in_chunks(jp2, indices)

        # combine the chunks back into one numpy array
        im = concatenate_chunks(ims, self.args['block_read'])

        # get roi
        x_ind = self.args["x_ind"]
        y_ind = self.args["y_ind"]
        im_roi = im[y_ind:, x_ind:]

        # downsample
        if self.args['downsample'] != 1.0:
            im_roi = downsample(im_roi, self.args['downsample'])

        # convert to greyscale
        im_grayscale = cv2.cvtColor(im_roi, cv2.COLOR_RGB2GRAY)

        tifffile.imwrite(self.args['tiff_output_file'], im_grayscale)
