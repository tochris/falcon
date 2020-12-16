import matplotlib.pyplot as plt
import csv
import pickle
import os

def save_line_chart(savefolder, filename, y_plot_list, x_plot_list=None,
                    diag_title='Titel', xlabel='x', ylabel='y', pyplotBool=True,
                    write_to_txtfile_Bool=True):
    """Saves a line chart png file defined by y_plot_list and x_plot_list
    and saves two txt files for x and y values
    """
    #plot diagram
    if pyplotBool == True:
        f1 = plt.figure()
        plt.title(diag_title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if x_plot_list == None:
            plt.plot(y_plot_list)
        elif x_plot_list != None:
            plt.plot(x_plot_list, y_plot_list)
        f1.savefig(savefolder + filename + '.png', dpi=200, bbox_inches = "tight")
        plt.close()
    #write to txt file
    if write_to_txtfile_Bool == True:
        file1 = open(savefolder + filename + '.txt', 'w')
        for i in y_plot_list:
            file1.write(str(i)+'\n')
        file1.close()
        if x_plot_list != None:
            file1 = open(savefolder + filename + '_x_Axis.txt', 'w')
            for i in x_plot_list:
                file1.write(str(i)+'\n')
            file1.close()

def save_dict_to_structured_txt(savefolder, results_dict, filename):
    """Saves dictionary in a txt file where items in dictionary are
    seperated by rows
    """
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    file1 = open(savefolder + filename + '.txt', 'w')
    for key in results_dict.keys():
        file1.write(str(key)+':   '+str(results_dict[key])+'\n')
    file1.close()

def save_dict_to_txt(savefolder, dic, filename):
    """Saves a dictionary as txt file"""
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    f = open(savefolder + filename + '.txt', 'w')
    f.write(str(dic))
    f.close()

def save_dict_to_csv(savefolder, params_dict, filename):
    """Saves dictionary as csv and pickled file"""
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    writer = csv.writer(open(savefolder + filename + '.csv', 'w'))
    for key, val in params_dict.items():
        writer.writerow([key, val])

def save_dict_to_pkl(savefolder, dic_logits, filename):
    """Saves dict as pickled file"""
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    save_name = os.path.join(savefolder+ filename + '.pkl')
    with open(save_name, "wb") as f:
        pickle.dump(dic_logits, f)
    f.close()

def load_dict_from_pkl(savefolder, filename):
    """Loads dict from pickled file"""
    print(savefolder)
    print(filename)
    save_name = os.path.join(savefolder+ filename + '.pkl')
    with open(save_name, 'rb') as f:
        params = pickle.load(f)
    f.close()
    return params
