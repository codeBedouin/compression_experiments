import json
import bokeh
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show

file_list = ['1_bit_stats.json', '2_bit_stats.json', '3_bit_stats.json',
             '4_bit_stats.json', '5_bit_stats.json', '6_bit_stats.json',
             '7_bit_stats.json', 'all_bit_stats.json']
color_list = ["red", "blue", "green", "purple", "black", "orange", "pink",
              "olive"]
output_file("compression_experiments.html")
loss_plot = figure(width=1500, height=500, title="Loss vs steps",
                   x_axis_label="Step Number", y_axis_label="loss value")
accuracy_plot = figure(width=1500, height=500, title="Accuracy vs epochs",
                       x_axis_label = "Epoch Number", y_axis_label= "Accuracy")
compression_ratio = figure(width=500, height=500, 
                           title="Compression ratio vs Number of bits",
                           x_axis_label= "Digits after decimal",
                           y_axis_label="Compression Ratio")

compress_val_list = list()
for idx, f in enumerate(file_list):
    print (f)
    with open(f, 'r') as in_file:
        file_data = json.load(in_file)
        loss_vals = file_data.get('loss_value')
        accuracy_vals = file_data.get('accuracy')
        compression_vals = file_data.get('compression_ratio')
        x_axis = range(len(loss_vals))
        loss_plot.circle(x_axis, loss_vals, color=color_list[idx],
                         legend=f.rsplit('_', 1)[0])
        loss_plot.line(x_axis, loss_vals, line_color=color_list[idx],
                       legend=f.rsplit('_',1)[0])
        
        x_axis = range(len(accuracy_vals))
        accuracy_plot.circle(x_axis, accuracy_vals, color=color_list[idx],
                             legend=f.rsplit('_',1)[0])
        accuracy_plot.line(x_axis, accuracy_vals, line_color=color_list[idx],
                           legend=f.rsplit('_',1)[0])
        try:
            compress_avg = sum(compression_vals)/len(compression_vals)
        except:
            break
        compress_val_list.append(compress_avg)


compression_ratio.circle(range(len(compress_val_list)), compress_val_list)

p = column(loss_plot, accuracy_plot, compression_ratio)
show(p)


