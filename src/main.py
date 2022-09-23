from utils import gf2


if __name__ == '__main__':
    # gf2 = r"D:\gf2_py6s\GF2_PMS2_E132.3_N33.5_20170422_L1A0002321961.tar.gz"
    # 可以输入文件下，下面全是高分
    gf2_path = r"/gf2/zhengzhou/GF2_PMS2_E113.6_N34.7_20201231_L1A0005358712.tar.gz"

    out = r"gf2_preprocess/gf2_preprocessing/src"
    dem = r"data_fuzhu/dem/ASTGTM2_N34E113_dem.tif"
    ref = r"data_fuzhu/ref_high_res_ge/zhengzhou_gf_googleearth_resample.tif"

    pre = gf2(gf2_path, out, ref, dem)
    pre.init_path()
    pre.correct()
    pre.target_output([3, 2, 1])