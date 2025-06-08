import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap

def draw_box(ax, center_x, center_y, width, height, text, facecolor, text_color='white', **kwargs):
    """
    Draws a rounded rectangle with centered text.
    This robust version safely handles keyword arguments like 'alpha', 'edgecolor', and 'linewidth'
    by popping them from kwargs and using a default value if not provided.
    This prevents TypeError for multiple values.
    """
    # Set default values that can be overridden by keyword arguments
    alpha = kwargs.pop('alpha', 0.9)
    edgecolor = kwargs.pop('edgecolor', 'black')
    linewidth = kwargs.pop('linewidth', 1.5)
    
    box = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2), 
        width, height,
        boxstyle='round,pad=0.1',
        facecolor=facecolor,
        edgecolor=edgecolor,  # Use the safe edgecolor value
        alpha=alpha,          # Use the safe alpha value
        linewidth=linewidth,  # Use the safe linewidth value
        **kwargs              # Pass any other remaining keywords
    )
    ax.add_patch(box)
    ax.text(
        center_x, center_y, text, 
        ha='center', va='center', 
        fontsize=12, fontweight='bold', color=text_color,
        wrap=True
    )
    return box

# --- Main Plotting Script ---
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(-2, 14)
ax.axis('off')

# --- Colors ---
real_color = '#2E8B57'
artificial_color = '#4169E1'
process_color = '#FF6347'
model_color = '#9370DB'
eval_color = '#FFD700'
generation_color = '#FF8C00'
artifact_color = '#00B3B3'

# --- Title ---
ax.text(10, 13, 'Pipeline Alur Data Deteksi Kecurangan', fontsize=26, fontweight='bold', ha='center')

# === 1. DATA SOURCES LEVEL ===
# Custom larger text for important data sources
box1 = FancyBboxPatch((4-2, 11.5-0.75), 4, 1.5, boxstyle='round,pad=0.1', facecolor=real_color, edgecolor='black', alpha=0.9, linewidth=1.5)
ax.add_patch(box1)
ax.text(4, 11.5, 'Data Riil ITF\n446,720 events\n5,562 users', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

box2 = FancyBboxPatch((16.5-1.5, 12.2-0.4), 3, 0.8, boxstyle='round,pad=0.1', facecolor=generation_color, edgecolor='black', alpha=0.9, linewidth=1.5)
ax.add_patch(box2)
ax.text(16.5, 12.2, 'Data Generator\nStochastic + Rule-based', ha='center', va='center', fontsize=13, fontweight='bold', color='white')

box3 = FancyBboxPatch((16.5-1.5, 10.8-0.65), 3, 1.3, boxstyle='round,pad=0.1', facecolor=artificial_color, edgecolor='black', alpha=0.9, linewidth=1.5)
ax.add_patch(box3)
ax.text(16.5, 10.8, 'Data Artifisial\n800 samples\n+ Ground Truth', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.add_patch(FancyArrowPatch((16.5, 11.7), (16.5, 11.4), mutation_scale=20, arrowstyle='-|>', color=generation_color, linewidth=2))

# === 2. PREPROCESSING PIPELINE ===
pipeline_bg = FancyBboxPatch((1.5, 6), 17, 4, boxstyle='round,pad=0.2', facecolor='whitesmoke', edgecolor='gray', linewidth=1.5, ls='--')
ax.add_patch(pipeline_bg)
ax.text(10, 9.5, 'Preprocessing Pipeline', ha='center', va='center', fontsize=20, fontweight='bold', color='black')

# Mode indicators (these calls pass 'alpha' and now work correctly)
draw_box(ax, 5, 8.8, 4, 0.6, 'Detection Mode', real_color, alpha=0.3, text_color=real_color)
draw_box(ax, 14.5, 8.8, 3, 0.6, 'Training Mode', artificial_color, alpha=0.3, text_color=artificial_color)

# Preprocessing modules (these calls pass 'edgecolor' and now work correctly)
module_positions = {'Load': 4, 'Preproc': 8, 'FeatureEng': 12, 'VIF': 16}
draw_box(ax, module_positions['Load'], 7.2, 3, 1.4, 'Data Loading', 'white', text_color='black', edgecolor=process_color)
draw_box(ax, module_positions['Preproc'], 7.2, 3, 1.4, 'Core Preprocessing', 'white', text_color='black', edgecolor=process_color)
draw_box(ax, module_positions['FeatureEng'], 7.2, 3, 1.4, 'Feature Engineering', 'white', text_color='black', edgecolor=process_color)
draw_box(ax, module_positions['VIF'], 7.2, 3, 1.4, 'VIF Analysis\n& Scaling', 'white', text_color='black', edgecolor=process_color)

# Arrows between modules
for start, end in [('Load', 'Preproc'), ('Preproc', 'FeatureEng'), ('FeatureEng', 'VIF')]:
    ax.add_patch(FancyArrowPatch((module_positions[start] + 1.5, 7.2), (module_positions[end] - 1.5, 7.2), mutation_scale=20, arrowstyle='-|>', color=process_color, linewidth=2))

ax.text(17.5, 7.2, '35 → 8\nfitur', ha='center', va='center', fontsize=13, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='gold', alpha=0.8, edgecolor='orange'))

# === 3. OUTPUTS & ARTIFACTS LEVEL ===
# Custom larger text for output boxes
output1 = FancyBboxPatch((5-2.75, 3.8-1.1), 5.5, 2.2, boxstyle='round,pad=0.1', facecolor=eval_color, edgecolor='black', alpha=0.9, linewidth=1.5)
ax.add_patch(output1)
ax.text(5, 3.8, 'Detection Results\n131,479 deteksi\n4,093 users affected\n29.43% detection rate', ha='center', va='center', fontsize=13, fontweight='bold', color='black')

output2 = FancyBboxPatch((15-2.75, 3.8-1.1), 5.5, 2.2, boxstyle='round,pad=0.1', facecolor=model_color, edgecolor='black', alpha=0.9, linewidth=1.5)
ax.add_patch(output2)
ax.text(15, 3.8, 'Model Training & Validation\nRandom Forest • SVM\nNeural Network • Gradient Boosting\nEnsemble Architecture', ha='center', va='center', fontsize=13, fontweight='bold', color='white')

output3 = FancyBboxPatch((10-2.5, 1.2-0.75), 5, 1.5, boxstyle='round,pad=0.1', facecolor=artifact_color, edgecolor='black', alpha=0.9, linewidth=1.5)
ax.add_patch(output3)
ax.text(10, 1.2, 'Saved Artifacts\nStandardScaler\nFeature Selector\nSimpleImputer', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# === 4. FLOW ARROWS ===
ax.add_patch(FancyArrowPatch((4, 10.7), (4, 8.6), connectionstyle="arc3,rad=-0.2", mutation_scale=25, arrowstyle='-|>', color=real_color, linewidth=2.5))
ax.add_patch(FancyArrowPatch((16.5, 10.1), (14.5, 8.6), connectionstyle="arc3,rad=0.4", mutation_scale=25, arrowstyle='-|>', color=artificial_color, linewidth=2.5))
ax.add_patch(FancyArrowPatch((5, 6.0), (5, 4.9), connectionstyle="arc3,rad=-0.2", mutation_scale=25, arrowstyle='-|>', color=real_color, linewidth=2.5))
ax.add_patch(FancyArrowPatch((14.5, 6.0), (15, 4.9), connectionstyle="arc3,rad=-0.2", mutation_scale=25, arrowstyle='-|>', color=artificial_color, linewidth=2.5))
ax.add_patch(FancyArrowPatch((14.5, 6.0), (11.5, 2.4), connectionstyle="arc3,rad=0.4", mutation_scale=20, arrowstyle='-|>', color='gray', linewidth=2, linestyle='--'))
ax.add_patch(FancyArrowPatch((8.5, 2.4), (5, 6.0), connectionstyle="arc3,rad=0.4", mutation_scale=20, arrowstyle='-|>', color='gray', linewidth=2, linestyle='--'))

# === 5. KEY DIFFERENCES & LEGEND ===
key_text = '★ Training Mode: Fit preprocessing & save artifacts | ★ Detection Mode: Load & apply artifacts | ★ VIF reduces 35→8 features | ★ Ground truth only for artificial data'
wrapped_text = textwrap.fill(key_text, width=140)
# Custom larger text for key information
key_box = FancyBboxPatch((10-9, -0.8-0.6), 18, 1.2, boxstyle='round,pad=0.1', facecolor='lightyellow', edgecolor='orange', alpha=0.9, linewidth=2)
ax.add_patch(key_box)
ax.text(10, -0.8, wrapped_text, ha='center', va='center', fontsize=12, fontweight='bold', color='black')

legend_elements = [
    mpatches.Patch(color=real_color, label='Data Riil & Detection Path'),
    mpatches.Patch(color=artificial_color, label='Data Artifisial & Training Path'),
    mpatches.Patch(color=generation_color, label='Data Generation Process'),
    mpatches.Patch(facecolor='white', edgecolor=process_color, hatch='xx', label='Preprocessing Step'),
    mpatches.Patch(color=model_color, label='Model Training'),
    mpatches.Patch(color=eval_color, label='Detection Results'),
    mpatches.Patch(color=artifact_color, label='Saved Artifacts'),
    mpatches.Patch(facecolor='none', edgecolor='gray', ls='--', label='Artifact Flow (Save/Load)')
]
ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.01, 0.01), fontsize=13, framealpha=0.95)

# === 6. SAVE FIGURE ===
plt.tight_layout(pad=0.5)
plt.savefig('technical_pipeline_flow_final.pdf', bbox_inches='tight', facecolor='white')

print("✅ Successfully saved the diagram as 'technical_pipeline_flow_final.pdf'")