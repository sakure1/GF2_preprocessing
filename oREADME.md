# GF2_preprocessing
* 正射校正
* 图像配准
* 图像融合


# Env
1.  Anaconda 创建新环境 (推荐)
``` shell
$ conda create -c conda-forge --name arosics python=3
$ conda activate arosics
$ conda install -c conda-forge 'arosics>=1.3.0'
```

2. 或者也可以在需要的环境里安装以下依赖  conda install
``` shell
* cartopy
* gdal
* geopandas
* matplotlib
* numpy
* pandas
* pyfftw <0.13.0
* pykrige
* pyproj >2.2.0
* scikit-image >=0.16.0
* shapely
* arosics
```

# 用法
``` shell
from utils import gf2

if __name__ == '__main__':
    # 直接输入高分2压缩文件，或者输入压缩文件所在的上级目录
    gf2_path = r"/gf2/zhengzhou/GF2_PMS2_E113.6_N34.7_20201231_L1A0005358712.tar.gz"
    
    # 结果输出路径
    out = r"gf2_preprocess/gf2_preprocessing/src"
    # 高分影像范围dem数据
    dem = r"data_fuzhu/dem/ASTGTM2_N34E113_dem.tif"
    # 图像配准用数据
    ref = r"data_fuzhu/ref_high_res_ge/zhengzhou_gf_googleearth_resample.tif"

    pre = gf2(gf2_path, out, ref, dem)
    pre.init_path()
    pre.correct()
    # 指定波段输出tif影像
    pre.target_output([3, 2, 1])
