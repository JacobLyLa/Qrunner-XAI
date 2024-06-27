import matplotlib.pyplot as plt

def set_size_and_get_figsize(num_in_row=1):
    # if 1: font is equivalent for all plots
    # if less than 1: font size slightly decreases with number of plots
    scale_pow = 0.5 
    font_size = 8 # probably 10, but most labels are bold
    resolution = 2 # so latex scales down, makes image quality better
    aspect_ratio = 16 / 9
    text_width_in = resolution*350 / 73
    plot_height_in = text_width_in / aspect_ratio

    #global_font_size = resolution * font_size * num_in_row * num_in_row ** scale_pow
    global_font_size = resolution * font_size * num_in_row ** scale_pow
    legend_font_size = global_font_size * 0.7

    plt.rcParams.update({
        'font.size': global_font_size,
        'legend.fontsize': legend_font_size,
        'font.family': 'serif'
    })
    return text_width_in, plot_height_in