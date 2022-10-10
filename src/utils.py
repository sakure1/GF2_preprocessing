"""
GF2 pre-processing workflow
utils.py
functions of pre-processing
"""
import os
from osgeo import gdal, osr
import logging
from arosics import COREG
from glob import glob
import tarfile
from tqdm import tqdm
import numpy as np
from arosics import COREG_LOCAL, DESHIFTER
import shutil
from pansharpen import gdal_pansharpen

logging.getLogger().setLevel('INFO')
gdal.AllRegister()


class gf2:
    def __init__(self,
                 gf2_path:str,
                 out_dir:str,
                 ref_image:str,
                 dem:str,
                 pan_res=8e-06,
                 mul_res=3.2e-05,
                 correct_type = 0,    # 0 means global,   1 means local
                 cpus = None          # 防止bug
                 ):
        self.gf2 = gf2_path
        self.save = out_dir
        self.ref = ref_image
        self.dem = dem
        self.pan_res = pan_res
        self.mul_res = mul_res
        self.pan_path = None
        self.mss_path = None
        self._dir = []
        # Intermediate process files
        self.garbage = []
        self.correct_files = []

    def unpackage(self, file_name):
        # 提取解压文件夹名
        if ".tar.gz" in file_name:
            out_dir = file_name.split(".tar.gz")[0]
        else:
            out_dir = file_name.split(".")[0]
        # 进行解压
        with tarfile.open(file_name) as file:
            info = "===untar {}".format(file_name)
            logging.info(info)
            file.extractall(path=out_dir)
        return out_dir

    def __ortho(self, file_name, res, out_name, epsg=4326):
        """
        正射校正啦
        :param file_name:
        :param res:
        :param out_name:
        :param epsg:
        :return:
        """
        info = "===正射校正(ortho): \n{} starting".format(file_name)
        logging.info(info)
        dataset = gdal.Open(file_name, gdal.GA_ReadOnly)

        dstSRS = osr.SpatialReference()
        dstSRS.ImportFromEPSG(epsg)

        tmp_ds = gdal.Warp(out_name, dataset, format='GTiff',
                           xRes=res, yRes=res, dstSRS=dstSRS,
                           rpc=True, resampleAlg=gdal.GRIORA_Bilinear,
                           transformerOptions=["RPC_DEM=" + self.dem])
        dataset = tds = None
        logging.info("===完成(Done)!")

    def __sharpen(self, pan_path, mul_path, out_path):
        info = "===数据融合(PanSharpen.): \n" \
               "PAN:{0} " \
               "MSS:{1}\n" \
               "starting".format(pan_path, mul_path)
        logging.info(info)
        gdal_pansharpen(["pass", pan_path, mul_path, out_path])
        logging.info("===完成(Done)!")

    def __regis(self, image, out, datatype=0, correct_type=1):
        """
        registration.
        :param image:
        :param out:
        :return:
        """
        info = "===图像配准(Registration.): \n{} starting".format(image)
        logging.info(info)
        result_path = []

        if datatype == 0:
            # PAN
            regis_image = image
        elif datatype == 1:
            #MSS
            regis_image = self.__tif_split(image, self.gf2)
        else:
            raise Exception("datatype error!")

        if correct_type == 0:
            if isinstance(regis_image, str):
                CR = COREG(self.ref, regis_image, ws=(1024, 1024), max_shift=200,
                           max_iter=10000, path_out=out)
                CR.correct_shifts()

            else:
                CR = COREG(self.ref, regis_image[0], ws=(1024, 1024), max_shift=200,
                           max_iter=10000)
                CR.calculate_spatial_shifts()
                for image in regis_image:
                    new_list = image.replace('.tif', '_regis.tif')
                    result_path.append(new_list)
                    self.garbage.append(new_list)
                    DESHIFTER(image, CR.coreg_info, path_out=new_list).correct_shifts()
                self.__concate_tif(result_path, out)

        elif correct_type == 1:
            if isinstance(regis_image, str):
                CR = COREG_LOCAL(self.ref, regis_image, 400, window_size=(512, 512),
                                 max_shift=200, max_iter=1000, path_out=out,
                                 # CPUs=1
                                 )
                CR.correct_shifts()
            else:
                CR = COREG_LOCAL(self.ref, regis_image[0], 200, window_size=(256, 256),
                                 max_shift=200, max_iter=1000,
                                 # CPUs=1
                                 )
                for image in regis_image:
                    new_list = image.replace('.tif', '_regis.tif')
                    result_path.append(new_list)
                    self.garbage.append(new_list)
                    DESHIFTER(image, CR.coreg_info, path_out=new_list).correct_shifts()
                self.__concate_tif(result_path, out)

        else:
            raise Exception("Error correct_type!! \n"
                            "correct_type must be 0 or 1 \n"
                            "0: global regif; 1: local regis!")

        # 配准输出

        del image
        del out
        del CR
        logging.info("===完成(Done)!")

    def init_path(self):
        if os.path.isdir(self.gf2):
            # files = os.listdir(self.gf2)
            zipfiles = glob(self.gf2+"/*.tar.gz")
            if len(zipfiles) == 0:
                self._dir.append(self.gf2)
            else:
                for zipfile in zipfiles:
                    self._dir.append(self.unpackage(zipfile))

    def correct(self):
        if len(self._dir) == 0:
            raise Exception("Please run init_path() before run()\n"
                            "if you have did it.\n"
                            "Maybe check whether input GF2 path is correct.\n")
        for gf2_dir in self._dir:
            logging.info(gf2_dir)
            print(self._dir[0])
            pan_path = glob(gf2_dir + "/*PAN*.tiff")[0]
            mss_path = glob(gf2_dir + "/*MSS*.tiff")[0]

            # step 1 正射校正
            pan_path_orth = pan_path.replace(".tiff", "_ortho.tiff")
            mss_path_orth = mss_path.replace(".tiff", "_ortho.tiff")
            self.__ortho(pan_path, self.pan_res, pan_path_orth)
            self.__ortho(mss_path, self.mul_res, mss_path_orth)

            self.garbage.append(pan_path_orth)
            self.garbage.append(mss_path_orth)

            # step 2 图像配准
            pan_path_orth_reg = pan_path_orth.replace(".tiff", "_regis.tiff")
            mss_path_orth_reg = mss_path_orth.replace(".tiff", "_regis.tiff")
            
            self.__regis(mss_path_orth, mss_path_orth_reg, datatype=1)
            self.__regis(pan_path_orth, pan_path_orth_reg)

            self.garbage.append(pan_path_orth_reg)
            self.garbage.append(mss_path_orth_reg)

            # step 3 图像融合
            gf2_orth_reg_sharp = pan_path_orth_reg.split("PAN")[0] + "_orth_regis_sharp.tiff"
            filename = os.path.basename(gf2_orth_reg_sharp)
            os.makedirs(self.save, exist_ok=True)
            save_path = os.path.join(
                self.save,
                filename
            )
            self.__sharpen(pan_path_orth_reg, mss_path_orth_reg, save_path)
            self.correct_files.append(save_path)
        for file in self.garbage:
            os.remove(file)

    def target_output(self, band_orders:list):
        """
        gf2 mss bangd parameters:
        band1: 0.45~0.52 um      Blue   蓝光
        band2: 0.52~0.59 um      Green  绿光
        band3: 0.63~0.69 um      Red    红光
        band4: 0.77~0.89 um      NIR   近红外
        :return:
        """
        info = "===Pre-process workflow done!\n" \
               "Now input band No.{} \n" \
               "to dir {} \n".format(band_orders, self.save)
        logging.info(info)
        assert len(band_orders) != 0, "band_orders length can not be Zero!"
        band_order_str = '_band'
        for order in band_orders:
            if int(order) == 0:
                raise Exception("\nband_orders include Zero, band index must in range(1, bands_num) \n")
            band_order_str = band_order_str + str(order)
        band_order_str = band_order_str + '.tiff'

        for index, correct_file in enumerate(self.correct_files):
            save_name = correct_file.replace(".tiff", band_order_str)


            # get tiff information
            tiff_gdal = self._read_tif(correct_file)
            width = tiff_gdal.RasterXSize
            height = tiff_gdal.RasterYSize
            GeoTrans = tiff_gdal.GetGeoTransform()
            GeoProj = tiff_gdal.GetProjection()

            channel = len(band_orders)

            # 写入
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(save_name, int(width), int(height), int(channel), gdal.GDT_Byte)
            info = "\n EXPORT to {} ".format(save_name)
            logging.info(info)
            if dataset is not None:
                dataset.SetGeoTransform(GeoTrans)  # 写入仿射变换参数
                dataset.SetProjection(GeoProj)  # 写入投影
                for index_, order in enumerate(band_orders):
                    single_band = self.__nornal(self.__get_single_band(tiff_gdal, order).ReadAsArray().astype(float))
                    dataset.GetRasterBand(index_+1).WriteArray(single_band)
            else:
                logging.info("{} save error".format(save_name))

    def __get_single_band(self, gdal_dataset, index:int):
        """
        index: start from 1, not 0
        :param gdal_dataset:
        :param index:
        :return:
        """
        return gdal_dataset.GetRasterBand(index)

    def writeTiff(self, im_data, im_geotrans, im_proj, path):
        """
        :param im_data: 需要保存的数组
        :param im_geotrans:
        :param im_proj:
        :param path: 输出路径
        :return:
        """
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)

        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        else:
            print("保存失败！")

        del dataset
        del driver

    def _read_tif(self, image_path):
        image_dataset = gdal.Open(image_path)
        assert image_dataset is not None, "{} open error!".format(image_path)
        return image_dataset

    def __nornal(self, image_array, type=np.uint8):
        image_array *= 255.0 / image_array.max()
        return image_array.astype(type)

    def __tif_split(self, image_path, out_dir):
        tiff_gdal = self._read_tif(image_path)
        GeoTrans = tiff_gdal.GetGeoTransform()
        GeoProj = tiff_gdal.GetProjection()
        channels = tiff_gdal.RasterCount

        split_lists = []

        for order in range(channels):
            single_band = self.__get_single_band(tiff_gdal, order + 1).ReadAsArray().astype(float)
            save_path = os.path.join(out_dir, 'band_' + str(order) + '.tif')
            split_lists.append(save_path)
            self.garbage.append(save_path)
            self.writeTiff(single_band, GeoTrans, GeoProj, save_path)
            del single_band
        del tiff_gdal
        return split_lists

    def __concate_tif(self, tif_lists, out_path):
        information = gdal.Open(tif_lists[0])
        width = information.RasterXSize
        height = information.RasterYSize
        geotrans = information.GetGeoTransform()
        projection = information.GetProjection()

        # dtype = np.int16
        channel = len(tif_lists)

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(out_path, int(width), int(height), int(channel), gdal.GDT_UInt16)

        if (dataset != None):
            dataset.SetGeoTransform(geotrans)  # 写入仿射变换参数
            dataset.SetProjection(projection)  # 写入投影
            for i in range(channel):
                information = gdal.Open(tif_lists[i])
                old_array = information.ReadAsArray(0, 0, width, height)
                dataset.GetRasterBand(i + 1).WriteArray(old_array)
        else:
            print("保存失败！")


if __name__ == '__main__':
    # # gf2 = r"D:\gf2_py6s\GF2_PMS2_E132.3_N33.5_20170422_L1A0002321961.tar.gz"
    # gf2_path = r"E:\python\DL_ZX\GF2_building\zhengzhou\zhengzhou"
    # out = r"F:\preprocess_zhengzhou_test"
    # dem = r"E:\python\DL_ZX\GF2_building\gf2\N34E113\ASTGTM2_N34E113\ASTGTM2_N34E113_dem.tif"
    # ref = r"F:\zhengzhou_gf_googleearth_resample.tif"
    #
    # pre = gf2(gf2_path, out, ref, dem)
    # pre.init_path()
    # pre.correct()
    # pre.target_output([3, 2, 1])
    #

    gf2_path = r"/home/xiaxiaoyun/yimin/data_fuzhu/gf2/zhengzhou"
    out = r"/home/xiaxiaoyun/yimin/data_fuzhu"
    ref = r"/home/xiaxiaoyun/yimin/data_fuzhu/ref_high_res_ge/zhengzhou_gf_googleearth_resample.tif"
    dem = r"/home/xiaxiaoyun/yimin/data_fuzhu/dem/ASTGTM2_N34E113_dem.tif"


    pre = gf2(gf2_path, out, ref, dem)
    pre.init_path()
    pre.correct()
    pre.target_output([3, 2, 1])












