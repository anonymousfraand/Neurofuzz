import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
# Generate universe variables
#   * Quality and service on subjective ranges [0, 10]
#   * Tip has a range of [0, 25] in units of percentage points
def tip(qulity,service):
    x_qual = np.arange(0, 11, 1)
    x_serv = np.arange(0, 11, 1)
    x_tip  = np.arange(0, 26, 1)
    # Generate fuzzy membership functions
    qual_lo = fuzz.trimf(x_qual, [0, 0, 5])
    qual_md = fuzz.trimf(x_qual, [0, 5, 10])
    qual_hi = fuzz.trimf(x_qual, [5, 10, 10])
    serv_lo = fuzz.trimf(x_serv, [0, 0, 5])
    serv_md = fuzz.trimf(x_serv, [0, 5, 10])
    serv_hi = fuzz.trimf(x_serv, [5, 10, 10])
    tip_lo = fuzz.trimf(x_tip, [0, 0, 13])
    tip_md = fuzz.trimf(x_tip, [0, 13, 25])
    tip_hi = fuzz.trimf(x_tip, [13, 25, 25])
    qual_level_lo = fuzz.interp_membership(x_qual, qual_lo,qulity)
    qual_level_md = fuzz.interp_membership(x_qual, qual_md, qulity)
    qual_level_hi = fuzz.interp_membership(x_qual, qual_hi, qulity)
    serv_level_lo = fuzz.interp_membership(x_serv, serv_lo, service)
    serv_level_md = fuzz.interp_membership(x_serv, serv_md, service)
    serv_level_hi = fuzz.interp_membership(x_serv, serv_hi, service)
    # Now we take our rules and apply them. Rule 1 concerns bad food OR service.
    # The OR operator means we take the maximum of these two.
    active_rule1 = np.fmax(qual_level_lo, serv_level_lo)
    # Now we apply this by clipping the top off the corresponding output
    # membership function with `np.fmin`
    tip_activation_lo = np.fmin(active_rule1, tip_lo)  # removed entirely to 0
    # For rule 2 we connect acceptable service to medium tipping
    tip_activation_md = np.fmin(serv_level_md, tip_md)
    # For rule 3 we connect high service OR high food with high tipping
    active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
    tip_activation_hi = np.fmin(active_rule3, tip_hi)
    tip0 = np.zeros_like(x_tip)
    # Aggregate all three output membership functions together
    aggregated = np.fmax(tip_activation_lo,
                         np.fmax(tip_activation_md, tip_activation_hi))
    # Calculate defuzzified result
    tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
    tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)  # for plot
    # Visualize this
    # Visualize these universes and membership functions
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(nrows=4, figsize=(8, 9))
    ax0.plot(x_qual, qual_lo, 'b', linewidth=1.5, label='Bad')
    ax0.plot(x_qual, qual_md, 'g', linewidth=1.5, label='Decent')
    ax0.plot(x_qual, qual_hi, 'r', linewidth=1.5, label='Great')
    ax0.set_title('Food quality')
    ax0.legend()
    ax1.plot(x_serv, serv_lo, 'b', linewidth=1.5, label='Poor')
    ax1.plot(x_serv, serv_md, 'g', linewidth=1.5, label='Acceptable')
    ax1.plot(x_serv, serv_hi, 'r', linewidth=1.5, label='Amazing')
    ax1.set_title('Service quality')
    ax1.legend()
    ax2.plot(x_tip, tip_lo, 'b', linewidth=1.5, label='Low')
    ax2.plot(x_tip, tip_md, 'g', linewidth=1.5, label='Medium')
    ax2.plot(x_tip, tip_hi, 'r', linewidth=1.5, label='High')
    ax2.set_title('Tip amount')
    ax2.legend()
    ax3.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
    ax3.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
    ax3.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
    ax3.fill_between(x_tip, tip0, aggregated, facecolor='Orange', alpha=0.7)
    ax3.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
    ax3.set_title('Aggregated membership and result (line)')
    ax3.legend()
    # Turn off top/right axes
    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()
    plt.savefig('output/out2.png')
    return tip
    # plt.show()
visualize.py :
import cv2
from tkinter import *
from tkinter import ttk, messagebox
from rpm import motor_rmp
from tip import tip
class visualize():
    def __init__(self):
        self.window = Tk()
        self.window.title("Fuzzy Neural Network Systems")
        self.window.geometry('800x950')
        self.tab_control = ttk.Notebook(self.window)
        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab1, text='RPM Calculate')
        self.tab_control.add(self.tab2, text='TIP Calculate')
        self.tab_control.add(self.tab3, text='Info')
        self.introframe = ttk.Labelframe(self.tab1, )
        self.info_1 = ttk.Labelframe(self.tab3,text= "Submited to")
        self.info_2 = ttk.Labelframe(self.tab3,text= "Submited by")
        self.l_frame = ttk.Labelframe(self.tab1, width=100, height=100)
        self.v_frame = ttk.Labelframe(self.tab1,text='visualize')
        heading = Label(self.introframe, text="Fuzzy Neural Network Systems\n Motor Control Using Mandani Method",font=("Arial", 12))
        heading.grid(column=0,row=0,sticky=N)
        submited_to = Label(self.info_1, text=" Prof. Young Im Cho\n Gachon University, South Korea",
                        font=("Arial", 12))
        submited_to.grid(column=0, row=1, sticky=N)
        submited_by = Label(self.info_2, text="Soikat Hasan Ahemd\n ID: 202040110 \nGachon University, South Korea",
                            font=("Arial", 12))
        submited_by.grid(column=0, row=1, sticky=N)
        Result_txt = Label(self.l_frame, text="Calculated RPM : ",font=("Arial", 14))
        Result_txt.grid(column=0, row=4,sticky=N)
        self.Result = Label(self.l_frame, text="N/A",font=("Arial", 16))
        self.Result.grid(column=1, row=4,sticky=N)
        input_txt = Label(self.l_frame, text="Insert Voltage\n (0~5) : ",font=("Arial", 14))
        input_txt.grid(column=0, row=2,sticky=N)
        self.txt = Entry(self.l_frame,width=10)
        self.txt.grid(column=1, row=2,sticky=N)
        btn = Button(self.l_frame, text="Calculate RPM", command=self.clicked)
        btn.grid(column=3, row=2,sticky=N)
 
        self.canvas = Canvas(self.v_frame, width=700, height=800)
        # canvas.pack()
        self.canvas.grid( row=0)
        #............................................................................................
        self.introframe2 = ttk.Labelframe(self.tab2, )
        self.l_frame2 = ttk.Labelframe(self.tab2, width=100, height=100)
        self.v_frame2 = ttk.Labelframe(self.tab2, text='visualize')
        heading2 = Label(self.introframe2, text="Fuzzy Neural Network Systems\n TIP Calculation Using Mandani Method",
                        font=("Arial", 12))
        heading2.grid(column=0, row=0, sticky=N)
        Result_txt2 = Label(self.l_frame2, text="Calculated TIP : ", font=("Arial", 14))
        Result_txt2.grid(column=0, row=4, sticky=N)
        self.Result2 = Label(self.l_frame2, text="N/A", font=("Arial", 16))
        self.Result2.grid(column=1, row=4, sticky=N)
        input_txt1 = Label(self.l_frame2, text="Insert service\n (0~10) : ", font=("Arial", 14))
        input_txt1.grid(column=0, row=2, sticky=N)
        self.txt11 = Entry(self.l_frame2, width=10)
        self.txt11.grid(column=1, row=2, sticky=N)
        input_txt_2 = Label(self.l_frame2, text="Insert Food Quality\n (0~10) : ", font=("Arial", 14))
        input_txt_2.grid(column=0, row=3, sticky=N)
        self.txt12 = Entry(self.l_frame2, width=10)
        self.txt12.grid(column=1, row=3, sticky=N)
        btn2 = Button(self.l_frame2, text="Calculate TIP", command=self.clickedtip)
        btn2.grid(column=3, row=3, sticky=N)
        self.canvas2 = Canvas(self.v_frame2, width=700, height=800)
        # canvas.pack()
        self.canvas2.grid(row=0)
        self.tab_control.pack(expand=1, fill='both')
        self.introframe.pack()
        self.introframe2.pack()
        self.info_1.pack()
        self.info_2.pack()
        self.l_frame.pack()
        self.l_frame2.pack()
        self.v_frame.pack()
        self.v_frame2.pack()
        self.window.mainloop()
    def clicked(self):
        input_txt = self.txt.get()
        if len(input_txt) == 0:
            messagebox.showwarning('Input Error', 'Please Input A Voltage (0~5)')
        else:
            try:
                value = float(input_txt)
                if value < 0.0 or value > 5.0:
                    messagebox.showwarning('Input Error', 'value Out of range.\n Input should be 0 ~ 5')
                else:
                    out = motor_rmp(value)
                    self.Result.configure(text='{:02f}'.format(out))
                    img = cv2.imread('output/out.png')
                    resize = cv2.resize(img,(600,700))
                    cv2.imwrite('output/out.png', resize)
                    self.img = PhotoImage(file="output/out.png")
                    self.canvas.create_image(20, 20, anchor=NW, image=self.img)
                    # self.canvas.configure(image=self.img)
            except:
                messagebox.showwarning('Input Error', 'Input Should be Integer or Float Value')
    def clickedtip(self):
        input_txt1 = self.txt11.get()
        input_txt2 = self.txt12.get()
        if len(input_txt1) == 0 or len(input_txt2) == 0 :
            messagebox.showwarning('Input Error', 'Input can not be empty')
        else:
            try:
                service = float(input_txt1)
                quality = float(input_txt2)
 
                if service < 0.0 or service > 10.0 or quality < 0.0 or quality > 10.0:
                    messagebox.showwarning('Input Error', 'value Out of range.\n Input should be 0 ~ 10')
                else:
                    out = tip(quality,service)
                    self.Result2.configure(text='{:02f}'.format(out))
                    img2 = cv2.imread('output/out2.png')
                    resize2 = cv2.resize(img2, (600, 700))
                    cv2.imwrite('output/out2.png', resize2)
                    self.img2 = PhotoImage(file="output/out2.png")
                    self.canvas2.create_image(20, 20, anchor=NW, image=self.img2)
            except:
                messagebox.showwarning('Input Error', 'Input Should be Integer or Float Value')
