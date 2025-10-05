import wrf
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm

# === 基本設定 ===
ncycles = 16
base_path = "./example/"
target_filename = "fcst_d03_2024-09-22_03_3A00_3A00"   # 取每個 cycle 的同一個檔案
threshold = 1.0  # mm

rain_total_list = []

# === 讀取每個 cycle ===
for ncycle in range(1, ncycles + 1):
    path = f"{base_path}cycle{ncycle:02d}/{target_filename}"
    print(f"Processing: {path}")
    
    # 讀檔
    ncfile = nc.Dataset(path)
    
    # 取出降水變數
    rainc = wrf.getvar(ncfile, "RAINC")
    rainnc = wrf.getvar(ncfile, "RAINNC")
    rain_total = rainc + rainnc
    
    # 若有時間維度，取最後一個時間步
    if len(rain_total.dims) == 3:
        rain_total = rain_total[-1, :, :]
    
    rain_total_list.append(rain_total)

# === 組成陣列 (members, lat, lon) ===
rain_total_array = np.array(rain_total_list)

# === 計算 PQPF ===
pqpf = np.sum(rain_total_array > threshold, axis=0) / len(rain_total_list)

# === 取經緯度 ===
lats, lons = wrf.latlon_coords(rain_total)

#COLORBAR
# 1. 定義顏色分界點 (共 6 個點，5 個色塊)
levels = np.linspace(0, 1, 6)
# 2. 取出 YlGnBu 的原始顏色，通常取 N-1 個 (這裡取 5 個)
#    我們使用 get_cmap 從 Matplotlib 獲取 YlGnBu 色圖。
original_cmap = cm.get_cmap('YlGnBu', 6) 
original_colors = original_cmap(np.arange(6))
# 3. 創建新的顏色列表：將第一個顏色 (黃色) 替換為白色
#    我們保留 YlGnBu 所有的深色，只將最淺的顏色換掉。
new_colors = np.copy(original_colors)
new_colors[0] = [1, 1, 1, 1]  # [R, G, B, Alpha] = [1, 1, 1, 1] 即白色
# 4. 建立新的 Colormap 物件
custom_cmap = ListedColormap(new_colors)

# === 畫圖 ===
plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

cf = plt.contourf(lons, lats, pqpf, 
                 levels=levels, 
                 cmap=custom_cmap, # 使用自定義的色圖
                 transform=ccrs.PlateCarree())

# 地形邊界
ax.coastlines(resolution='50m', color='black', linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

plt.title(f"PQPF (>{threshold} mm) Probability")
plt.colorbar(cf, label="Probability")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig("hw1_PQPF.png", dpi=150)
plt.show()
