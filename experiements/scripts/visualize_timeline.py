import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Data
data = [
    # Two-stage
    {"name": "R-CNN", "date": "2014-01", "type": "Two-stage"},
    {"name": "SPPNet", "date": "2015-01", "type": "Two-stage"},
    {"name": "Fast R-CNN", "date": "2015-04", "type": "Two-stage"},
    {"name": "Faster R-CNN", "date": "2015-08", "type": "Two-stage"},
    {"name": "Mask R-CNN", "date": "2017-03", "type": "Two-stage"},
    
    # One-stage
    {"name": "SSD", "date": "2016-01", "type": "One-stage"},
    {"name": "RetinaNet", "date": "2017-08", "type": "One-stage"},
    {"name": "YOLOv3", "date": "2018-04", "type": "One-stage"},
    {"name": "YOLOv5", "date": "2020-06", "type": "One-stage"},
    {"name": "YOLOv7", "date": "2022-07", "type": "One-stage"},
    {"name": "YOLOv8", "date": "2023-01", "type": "One-stage"},
    
    # Anchor-free / DETR-based (Using Anchor-free category as requested)
    {"name": "CornerNet", "date": "2018-08", "type": "Anchor-free"},
    {"name": "FCOS", "date": "2019-04", "type": "Anchor-free"},
    {"name": "DETR", "date": "2020-05", "type": "Anchor-free"},
    {"name": "Deformable DETR", "date": "2020-10", "type": "Anchor-free"},
    {"name": "Conditional DETR", "date": "2021-01", "type": "Anchor-free"}, # 2021
    {"name": "Anchor DETR", "date": "2021-8", "type": "Anchor-free"},    # 2021
    {"name": "DN-DETR", "date": "2022-03", "type": "Anchor-free"},       # 2022
    {"name": "Efficient DETR", "date": "2022-06", "type": "Anchor-free"}, # 2022
    {"name": "RT-DETR", "date": "2023-04", "type": "Anchor-free"},       # 2023
]

# Parse dates and organize
for item in data:
    item['date_obj'] = datetime.strptime(item['date'], "%Y-%m")

# Sort by date ensures the alternating layout works correctly across categories
data.sort(key=lambda x: x['date_obj'])

# Colors for types
colors = {
    "Two-stage": "#3498db",  # Blue
    "One-stage": "#e74c3c",  # Red
    "Anchor-free": "#2ecc71" # Green
}

# Create figure
fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)

# Set background color
fig.patch.set_facecolor('#f5f5f5')
ax.set_facecolor('#f5f5f5')

# Y-positions for categories: Anchor-free (Top), One-stage (Middle), Two-stage (Bottom)
y_map = {"Anchor-free": 3, "One-stage": 2, "Two-stage": 1}
labels = ["Two-stage", "One-stage", "Anchor-free"]

# Plot lines and points
for item in data:
    d = item['date_obj']
    y = y_map[item['type']]
    c = colors[item['type']]
    
    # Plot point - Increased size from 150 to 400
    ax.scatter(d, y, color=c, s=1200, zorder=3, edgecolors='white', linewidth=2)
    
    # Vertical line to axis (optional, maybe just visual separation is enough)
    # ax.plot([d, d], [0, y], color=c, alpha=0.3, linestyle='--', zorder=1)
    
    # Label layout optimization
    # Alternate labels above and below based on CATEGORY index to ensure alternation on the same line
    same_type_data = [x for x in data if x['type'] == item['type']]
    local_idx = same_type_data.index(item)
    
    # Base offset: Alternate based on order within the specific line (category)
    is_top = local_idx % 2 == 1 # Start with bottom for 0 (optional choice)
    
    # Specific adjustments for crowded areas (Manual overrides)
    if item['name'] == "YOLOv6": is_top = True
    if item['name'] == "YOLOv7": is_top = False
    if item['name'] == "Faster R-CNN": is_top = False
    
    # Increase offset for better separation
    base_offset = 50
    offset_y = base_offset if is_top else -base_offset
    
    # Additional horizontal stagger for very close items
    offset_x = 0
    if item['name'] == "Fast R-CNN": offset_x = -20
    if item['name'] == "Faster R-CNN": offset_x = 20
    if item['name'] == "YOLOv6": offset_x = -15
    if item['name'] == "YOLOv7": offset_x = 15
    
    connection_style = "angle,angleA=0,angleB=90,rad=10"
    
    ax.annotate(item['name'], 
                (d, y), 
                xytext=(offset_x, offset_y), 
                textcoords='offset points',
                ha='center', va='center',
                fontsize=11, 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=c, alpha=0.95, linewidth=1.5),
                arrowprops=dict(arrowstyle="-", 
                              connectionstyle=connection_style,
                              color=c, 
                              linewidth=1.5,
                              alpha=0.8))

    # Add date label below/above
    date_str = item['date']
    if len(date_str) > 4: # If it has month
        date_lbl = d.strftime("%Y-%m")
    else:
        date_lbl = d.strftime("%Y")
        
    # ax.text(d, y - (offset*0.5), date_lbl, ha='center', va='center', fontsize=8, color='#555')


# Aesthetics
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(labels, fontsize=24, fontweight='bold', color='#333')
ax.set_ylim(0.5, 3.8)

# Format X axis
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=24)

# Grid
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

# Removing spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['bottom'].set_color('#888')

# Title
plt.title("Object Detection Model Timeline", fontsize=30, fontweight='bold', pad=20, color='#333')

# Add legend manually for types
markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='', markersize=24) for color in colors.values()]
plt.legend(markers, colors.keys(), loc='upper left', frameon=False, fontsize=30)

# Save
output_path = '/home/fyb/mydir/rf-detr/experiements/visualizations/model_timeline.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Timeline saved to {output_path}")
