import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import netCDF4 as nc
import wrf
import os
import re

#=== 設定 cycle ===
ncycle = 1
path = f'./example/cycle{ncycle:02d}/'

# 找出該 cycle 下所有 fcst 檔
file_list = sorted([f for f in os.listdir(path) if f.startswith("fcst_d03")])

print(f"Cycle {ncycle:02d} 檔案清單：")
for f in file_list:
    print(" -", f)

#=== 迴圈讀取每個 fcst 檔 ===
for fn in file_list:
    print(f"\n>>> 處理 {fn}")
    ncfile = nc.Dataset(os.path.join(path, fn))

    # 從檔名擷取時間標籤，例如 "3A30"
    # time_label = re.search(r'(\d_A\d+)', fn)
    # time_label = time_label.group(1) if time_label else "unknown"

    parts = fn.split('_')
    time_label = parts[4] # 提取 '3A30' 或 '3A00'

    # === θe 剖面 ===
    thetae = wrf.getvar(ncfile, "theta_e")
    z = wrf.getvar(ncfile, "z")

    start_point = wrf.CoordPair(lat=22.7, lon=119.9)
    end_point   = wrf.CoordPair(lat=25.05, lon=121.3)
    thetae_vc = wrf.vertcross(thetae, z, wrfin=ncfile,
                              start_point=start_point,
                              end_point=end_point,
                              latlon=True, meta=True)
    terrain = wrf.interpline(wrf.getvar(ncfile,"HGT"),
                             wrfin=ncfile,
                             start_point=start_point,
                             end_point=end_point)

    lat_vc = [p.lat for p in wrf.to_np(thetae_vc.coords["xy_loc"])]
    lat2d, z2d = np.meshgrid(lat_vc, wrf.to_np(thetae_vc.coords["vertical"]))

    levels = np.arange(340, 385, 5)
    thetae_colors = ['#000080','#0064C8','#00BFFF','#ADFF2F','#FFFF00',
                     '#FFA500','#FF4500','#A00000','#700000']
    cmap_thetae = mcolors.ListedColormap(thetae_colors)
    norm_thetae = BoundaryNorm(levels, cmap_thetae.N)

    plt.figure(figsize=(10,6))
    cs = plt.contourf(lat2d, z2d, wrf.to_np(thetae_vc),
                      levels=levels, cmap=cmap_thetae, norm=norm_thetae, extend="both")
    plt.colorbar(cs, label="$\\theta_e$ (K)", ticks=levels[::2])
    plt.fill_between(lat_vc, 0, terrain, color="black")
    plt.title(f"Theta-E Cross Section ({time_label})")
    plt.xlabel("Latitude")
    plt.ylabel("Height (m)")
    plt.savefig(f"hw1_thetae_cross_{time_label}.png", dpi=150)
    plt.show()

    # === Divergence ===
    u = wrf.getvar(ncfile, "ua")
    v = wrf.getvar(ncfile, "va")
    p = wrf.getvar(ncfile, "pressure")
    dx = ncfile.DX
    dy = ncfile.DY

    du_dx = (u[:, :, 2:] - u[:, :, :-2]) / (2 * dx)
    dv_dy = (v[:, 2:, :] - v[:, :-2, :]) / (2 * dy)
    div = du_dx[:, 1:-1, :] + dv_dy[:, :, 1:-1]
    div_850 = wrf.interplevel(div, p[:, 1:-1, 1:-1], 850)

    lats, lons = wrf.latlon_coords(p)
    lats = lats[1:-1, 1:-1]
    lons = lons[1:-1, 1:-1]

    plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    levels = np.arange(-9, 10, 3)
    cf = ax.contourf(lons, lats, div_850*1e5, levels=levels,
                     cmap='coolwarm_r', extend='both')
    plt.colorbar(cf, label="Divergence $10^{-5}\ s^{-1}$", ticks=levels)
    ax.coastlines()
    ax.set_title(f"850 hPa Divergence ({time_label})")
    plt.savefig(f"hw1_divergence_{time_label}.png", dpi=150)
    plt.show()

    ncfile.close()

print("\n✅ 全部圖片已完成繪製！")
