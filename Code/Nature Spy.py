from kivy.lang import Builder
from kivy.app import App
from kivy.uix.video import Video
from kivy.config import Config
from kivymd.app import MDApp
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.color_definitions import colors
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.list import OneLineIconListItem, IconLeftWidget
import webbrowser
from kivy.clock import Clock
from kivymd.uix.navigationdrawer import MDNavigationDrawer
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFillRoundFlatIconButton
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.properties import StringProperty
import webview
import cv2
import os
import torch
from torchvision import models, transforms
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.core.clipboard import Clipboard
from PIL import Image
import torch_directml
from torch import nn
import ast
from kivy.uix.gridlayout import GridLayout
from PIL import Image
from kivy.uix.image import Image as i
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
import shutil
from tkinter import filedialog,Tk
import wikipedia 
from kivy.core.window import Window
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms
# ffpyplayer

# ML Model START 

dml = torch_directml.device()
device ="cpu"
print(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class ConvolutionalModel(nn.Module):
    def __init__(self, pre_trained_model, output_shape, in_features):
        super(ConvolutionalModel, self).__init__()
        self.pre = nn.Sequential(*list(pre_trained_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=in_features, out_features=output_shape)
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.classifier(x)
        return x

def load_model(file_path):
    pre_trained_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    model_load = ConvolutionalModel(pre_trained_model, output_shape=3, in_features=512)
    model_load.load_state_dict(torch.load(file_path))
    print("Model loaded successfully!")
    return model_load

TOP_N_PREDICTIONS=2
CONFIDENCE_THRESHOLD=0.9
def preprocess_image(file_path):
    image = Image.open(file_path)
    image = resize_image(image)
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor


def resize_image(image, target_size=(224, 224)):
    return image.resize(target_size, 3)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, r'Model\Firstlayerv3.pth')
# model_path = r'D:\VS CODE\PY\Kivy\Model\combinedvalidation.pth' 
model = load_model(model_path)
model.to(device)
model.eval()


def predict_image(input_tensor):
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_indices = torch.topk(probabilities, TOP_N_PREDICTIONS)
    confidence = confidence.cpu().numpy()
    predicted_indices = predicted_indices.cpu().numpy()

    # Check if the highest confidence is below the threshold
    if confidence[0][0] < CONFIDENCE_THRESHOLD:
        print(f"Model is not confident. Highest Confidence: {confidence[0]}")
        for i in range(TOP_N_PREDICTIONS):
            if predicted_indices[0][i]==0:
               Name="Animal"
            elif predicted_indices[0][i]==1:
                Name="Others"
            else:
                Name="Plants"
            print(f"Prediction {i + 1}: NAME={Name}, Confidence={confidence[0][i]}")

    return predicted_indices[0][0]

def predict_second_layer(Name,input_tensor, output_shape, file_path):
    print(Name)
    input_tensor = input_tensor.to(device)
    pre_trained_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    model2_load = ConvolutionalModel(pre_trained_model, output_shape=output_shape, in_features=2048)
    model2_load.load_state_dict(torch.load(file_path))
    model2_load.eval()

    with torch.no_grad():
        output = model2_load(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()

    print(f"Second Layer Predicted Index: {predicted_index}")
    return predicted_index

# ML Model END


class ROILabeler:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.clone = self.image.copy()
        self.rectangles = []
        self.current_rectangle = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.drawing = False
        self.window_name = 'ROI Selector (Press c to capture, r to reset triangle and esc to quit window)'
        self.current_roi_index = 1
        self.resized_image = self.image.copy()
        self.resize_factor = 0.7
        self.resize_image()

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def resize_image(self):
        self.resized_image = cv2.resize(self.image, None, fx=self.resize_factor, fy=self.resize_factor)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x, self.end_y = x, y
            self.current_rectangle = (self.start_x, self.start_y, self.end_x, self.end_y)
            self.rectangles.append(self.current_rectangle)
            cv2.rectangle(self.resized_image, (self.start_x, self.start_y), (self.end_x, self.end_y), (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.resized_image)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.resized_image.copy()
                cv2.rectangle(temp_image, (self.start_x, self.start_y), (x, y), (0, 255, 0), 2)
                cv2.imshow(self.window_name, temp_image)

    def run(self):
        while True:
            cv2.imshow(self.window_name, self.resized_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.image = self.clone.copy()
                self.rectangles = []
                self.resize_image()

            elif key == ord('c'):
                self.save_images()
                break

            elif key == 27:
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()
        print("Selected Rectangles:", self.rectangles)

    def save_images(self):
        for i, rect in enumerate(self.rectangles):
            x1, y1, x2, y2 = rect
            x1, y1, x2, y2 = int(x1 / self.resize_factor), int(y1 / self.resize_factor), int(x2 / self.resize_factor), int(y2 / self.resize_factor)
            roi = self.clone[y1:y2, x1:x2]

            roi_filename = r"temp_roi.jpg"
            cv2.imwrite(roi_filename, roi)
            print(f"Saved ROI as {roi_filename}")


def open_link(link):
    webbrowser.open(link)

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

KV = '''
sss:
    id: video_player
    source: 'op1.mp4'
    state: 'play'
    fullscreen: True
    bg: 'black'
    options: {'eos': 'stop', 'allow_stretch': True,'media': 'video'}
'''

class sss(Video):
    def __init__(self, **kwargs):
        super(sss, self).__init__(**kwargs)
        self.state = 'play'
        self.fullscreen = True 
        self.bg = 'black'
        Clock.schedule_once(self.switch_screen, 15) 

    def switch_screen(self, dt):
        app = App.get_running_app()
        app.root.current = 'main_screen'
        app.stop()

class NatureEyes(App):
    def build(self):
        # ico = r"og1.png"
        self.title = "Nature Spy"
        # self.icon = ico
        Window.borderless = True
        ss = Builder.load_string(KV)
        return ss

class MainApp(MDApp):
    bg = r"og11.png"
    ico = r"og1.png"
    image_source = StringProperty()
    name = StringProperty()
    info = StringProperty()
    img_path=""
    i = 0 
    roi_im = StringProperty()
    def build(self):
        self.title = "Nature Spy"
        self.icon = r"og1.png"
        Window.borderless = False
        self.theme_cls.primary_palette = "Teal"
        self.theme_cls.primary_hue = "A700"
        self.theme_cls.accent_palette = "Cyan"
        self.theme_cls.accent_hue = "A700"
        self.theme_cls.theme_style = "Dark" 
        self.theme_cls.colors = colors
        root = Builder.load_string('''
MDBoxLayout:
    orientation: "vertical"

    MDTopAppBar:
        title: "Nature Spy"
        left_action_items: [["menu", lambda x: nav_drawer.set_state("toggle")]]
        elevation: 10

    MDNavigationLayout:

        ScreenManager:
            id: screen_manager

            Screen:
                name: "Home"
                BoxLayout:
                    orientation: 'vertical'
                    TabbedPanel:
                        id:tabs
                        do_default_tab: False
                        tab_width: root.width/2-3
                        tab_indicator_anim: True
                        TabbedPanelItem:
                            text: 'Input'
                            disabled: False
                            color:'cyan'
                            background_color: '#a4f4f9'
                            RelativeLayout:
                                orientation: 'vertical'
                                canvas.before:
                                    Rectangle:
                                        pos: self.pos
                                        size: self.size
                                        source: app.bg
                                Label:
                                    pos_hint: {"center_x": 0.5, "center_y": 0.75}
                                    color:'black'
                                    text: '[Give your image here !]'
                                    bold:True
                                    font_size:"30sp"
                                MDFillRoundFlatIconButton:
                                    text: "Open Image"
                                    pos_hint: {"center_x": 0.5, "center_y": 0.6}
                                    on_release: app.open_img()
                                MDFillRoundFlatIconButton:
                                    text: "Roi Selction Process"
                                    pos_hint: {"center_x": 0.2, "center_y": 0.3}
                                    on_release: app.roi()
                                MDFillRoundFlatIconButton:
                                    text: "Automatic Selection Process (BETA)"
                                    pos_hint: {"center_x": 0.8, "center_y": 0.3}
                                    on_release: app.auto()
                        TabbedPanelItem:
                            text: 'Output'
                            id:output_tab
                            color:'cyan'
                            background_color: '#a4f4f9'
                            disabled: False
                            RelativeLayout:
                                size_hint_y: 1
                                canvas.before:
                                    Rectangle:
                                        pos: self.pos
                                        size: self.size
                                        source: app.bg
                                Label:
                                    text: '[Selected Portion Result on Image :]'
                                    color:'black'
                                    bold:True
                                    font_size:"25sp"
                                    pos_hint: {"center_x": 0.5, "center_y": 0.925}
                                BoxLayout:
                                    orientation: "vertical"
                                    size_hint: 0.5, 0.5
                                    padding: "2dp"
                                    spacing: "2dp"
                                    pos_hint: {"center_x": 0.2, "center_y": 0.575}
                                    Image:
                                        source: app.roi_im
                                Label:
                                    text: app.name
                                    color:'black'
                                    bold:True
                                    pos_hint: {"center_x": 0.5, "center_y": 0.25}
                                MDFloatingActionButton:
                                    icon: "content-copy"
                                    size_hint:(0.1,0.1)
                                    size:(20,20)
                                    pos_hint: {"center_x": 0.5,"center_y": 0.15}
                                    on_press: app.copy_to_clipboard()
                                MDFillRoundFlatIconButton:
                                    text: "Click To Ask More!"
                                    pos_hint: {"center_x": 0.75, "center_y": 0.1}
                                    on_release: app.bot_link()
                                RelativeLayout:
                                    size_hint_x:0.5
                                    size_hint_y:0.5
                                    orientation: 'vertical'
                                    pos_hint: {"center_x": 0.7, "center_y": 0.6}
                                    BoxLayout:
                                        size_hint_y: 1
                                        ScrollView:
                                            id: scroll_view3   
                                            MDLabel:
                                                text: app.info
                                                multiline:True
                                                halign:'center'
                                                bold:True
                                                color:'black'
                                MDFillRoundFlatIconButton:
                                    text: "Error Report"
                                    pos_hint: {"center_x": 0.2, "center_y": 0.1}
                                    height: "50dp"
                                    on_release: app.error()
            Screen:
                name: "Detected"
                BoxLayout:
                    orientation: 'vertical'
                    size_hint_y:1
                    TabbedPanel:
                        do_default_tab: False
                        tab_width: root.width/2-3
                        tab_indicator_anim: True
                        TabbedPanelItem:
                            text: 'Plants'
                            disabled: False
                            background_color: '#a4f4f9'
                            color:'cyan'
                            RelativeLayout:
                                size_hint_y: 1
                                canvas.before:
                                    Rectangle:
                                        pos: self.pos
                                        size: self.size
                                        source: app.bg
                                ScrollView:
                                    id: scroll_view
                                    ImageGrid:
                                        id: pl_grid
                        TabbedPanelItem:
                            text: 'Animals'
                            color:'cyan'
                            disabled: False
                            background_color: '#a4f4f9'
                            RelativeLayout:
                                size_hint_y: 1
                                canvas.before:
                                    Rectangle:
                                        pos: self.pos
                                        size: self.size
                                        source: app.bg
                                ScrollView:
                                    id: scroll_view2
                                    ImageGrid2:
                                        id: an_grid
            Screen:
                name: "AboutUs"
                BoxLayout:
                    orientation: 'vertical'
                    TabbedPanel:
                        do_default_tab: False
                        tab_width: root.width
                        tab_indicator_anim: True
                        TabbedPanelItem:
                            text: 'About Us'
                            disabled: False
                            color:'cyan'
                            background_color: '#a4f4f9'
                            RelativeLayout:
                                orientation: 'vertical'
                                canvas.before:
                                    Rectangle:
                                        pos: self.pos
                                        size: self.size
                                        source: app.bg
                                Image:
                                    source: app.ico
                                    pos_hint: {"center_x": 0.5, "center_y": 0.8}
                                    size_hint: None, None
                                    size: 600,600
                                    spacing: [0, 50]
                                Label:
                                    text: "Developer Name's : Strike, Night Wolf, Lucifer"
                                    font_size: "25sp"
                                    color: "black"
                                    bold:True
                                    pos_hint: {"center_x": 0.5, "center_y": 0.5}

                                Label:
                                    text: "Email: contact.nature.spy@gmail.com"
                                    font_size: "25sp"
                                    color: "black"
                                    bold:True
                                    pos_hint: {"center_x": 0.5, "center_y": 0.3}
                

        MDNavigationDrawer:
            id: nav_drawer

            BoxLayout:
                orientation: "vertical"
                Image:
                    source: app.ico
                MDList:
                    OneLineIconListItem:
                        text: "Home"
                        on_press:
                            nav_drawer.set_state("close")
                            screen_manager.current = "Home"
                        IconLeftWidget:
                            icon: "home"

                    OneLineIconListItem:
                        text: "Detected"
                        on_press:
                            nav_drawer.set_state("close")
                            screen_manager.current = "Detected"
                        IconLeftWidget:
                            icon: "google-lens"     

                    OneLineIconListItem:
                        text: "Nature Chat"
                        on_press:
                            nav_drawer.set_state("close")
                            app.bot_link()
                        IconLeftWidget:
                            icon: "chat"

                    OneLineIconListItem:
                        text: "About Us"
                        on_press:
                            nav_drawer.set_state("close")
                            screen_manager.current = "AboutUs"
                        IconLeftWidget:
                            icon: "information"

                    OneLineIconListItem:
                        text: "Feedback"
                        on_press:
                            nav_drawer.set_state("close")
                            app.feedback()
                        IconLeftWidget:
                            icon: "comment-quote"

                    OneLineIconListItem:
                        text: "Rate Us"
                        on_press:
                            nav_drawer.set_state("close")
                            app.rate_us_link("com.akm_appmakers")
                        IconLeftWidget:
                            icon: "star"
''') 
        return root
    
    def switch_tab(self):
        self.root.ids.tabs.switch_to(self.root.ids.output_tab)

    def rate_us_link(self,package_name):
        link = f"https://play.google.com/store/apps/details?id={package_name}"
        open_link(link)

    def feedback(self):
        link = "https://forms.gle/HLkbHMnBzFJ9JQWh7"
        open_link(link)

    def error(self):
        link = "https://forms.gle/Vo9qjkBhuovbapF17"
        open_link(link)

    def search(self):
        result = wikipedia.summary(self.name, sentences = 2) 
        self.info = result

    def copy_to_clipboard(self):
        Clipboard.copy(self.name)

    def search1(self,na):
        result = wikipedia.summary(na, sentences = 1) 
        return result 

    def open_img(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if(file_path==""):
            self.show_popup()
        else:
            self.img_path = file_path

        # popup = Popup(title="Select an Image", content=file_path, size_hint=(0.8, 0.8))
        # popup.open()

    def roi(self):
        if(self.img_path==""):
            self.show_popup()
        else:
            self.image_source = self.img_path
            roi_labeler = ROILabeler(self.image_source)
            roi_labeler.run()
            self.get_name()
            file = r"temp_roi.jpg"
            if os.path.exists(file):
                self.switch_tab()
                os.remove(r"temp_roi.jpg")
            self.switch_tab()
            self.img_path=""

    def auto(self):
        self.info = ""
        if(self.img_path==""):
            self.show_popup()
        else:
            self.image_source = self.img_path
            model_detection = self.load_detection_model()
            dic=self.process_image_and_classify_Auto_Detect(model_detection, self.image_source)
            self.name = str(dic)
            # self.info = "Auto Dection is in Beta Version So Now it doesn't Provide info for Detected labels, Please Get Information from our chat bot."
            self.switch_tab()
            l = dic.values()
            i=1
            a=1
            if(len(l)==0):
                self.info = "Auto detect is unable to Detect anyting, Please try with different image to detect Plant and Animals only."
            elif(len(l)<5 ):
                for j in l:
                    n=1
                    try:
                        while n:
                            a = self.search1(j)
                            self.info = self.info + f"{i}. {a}\n\n"
                            if(self.info==""):
                                n=1
                            else:
                                n=0
                    except Exception as e:
                        a=0
                        self.info = self.info + f"{i}. No Information Found Try our chat bot by Clicking on below button or Check your internet connection.\n\n"
                    i+=1
            else:
                self.info = "Labels Exceed the Display size limit, Please use our bot for Information"
            self.img_path=""
            if(a==0):
                self.show_popup2()

    def load_detection_model(self):
        model = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        model.to(device)
        model.eval()
        return model
        
    def classify_secondary_model(self,input_tensor, category,v):
        if v==0:
            path_model=os.path.join(current_directory, r'Model\animalsecondlayer.pth')
            path_model_text=os.path.join(current_directory, r'Model\animalsecondlayer.txt')
        else:
            path_model=os.path.join(current_directory, r'Model\secondplantfinal.pth')
            path_model_text=os.path.join(current_directory, r'Model\plantsecondlayer.txt')
        with open(path_model_text, 'r') as file:
            content = file.read()
            data_list = ast.literal_eval(content)
        pre_trained_model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        model2_load = ConvolutionalModel(pre_trained_model, output_shape=len(data_list), in_features=2048)
        model2_load.load_state_dict(torch.load(path_model))
        model2_load.eval()
        with torch.no_grad():
            output = model2_load(input_tensor)
        _, predicted_index = torch.max(output, 1)
        predicted_index_2 = predicted_index.item()
        # print(predicted_index_2)
        # print("HERE IS EXP HEREJBD")
        nam = data_list[predicted_index_2]
        pred_name = f'{category} / {nam}'
        # print(pred_name)
        return pred_name
    
    def classify_image(self,model, image):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = transform(Image.fromarray(image)).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output).item()
        return predicted_class

    def process_image_and_classify_Auto_Detect(self,model_detection,  image_path, confidence_threshold=0.40, min_box_size=100):#Crucial Parameter
        di={}
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        image_tensor.to(device)
        Marked_item=0
        with torch.no_grad():
            prediction = model_detection(image_tensor)

        boxes = prediction[0]['boxes'].cpu().numpy().astype(int)
        scores = prediction[0]['scores'].cpu().numpy()

        if len(boxes) == 0:
            print("No objects detected.")
            return

        # Filter boxes based on confidence threshold
        indices = [i for i, score in enumerate(scores) if score > confidence_threshold]

        # Apply Non-Maximum Suppression (NMS)
        #Nitesh iou_threshold seh tera box overlapping ka cover hota iska bhi chaiye toh slider kar sakta hai tu jitni value 0 ke close utna he kaam overlapping of boxes hoga
        keep_indices = nms(torch.tensor(boxes).float(), torch.tensor(scores), iou_threshold=0.45)#Crucial Parameter
        indices = [i for i in indices if i in keep_indices]

        # Store indices of boxes to ignore
        ignore_indices = set()

        for i in indices:
            if i in ignore_indices:
                continue
            box1 = boxes[i]
            score1 = scores[i]

            color = (0, 0, 255)
            # Calculate the width and height of the bounding box
            box_width = box1[2] - box1[0]
            box_height = box1[3] - box1[1]

            # Check if the box size is above the threshold
            if box_width * box_height < min_box_size:
                ignore_indices.add(i)
                continue

            # Check if the current box is completely contained by any other box
            for j in indices:
                if i != j and j not in ignore_indices:
                    box2 = boxes[j]

                    if self.is_box_contained(box2, box1):
                        ignore_indices.add(i)
                        break

            # Display bounding boxes only for detections with score above the threshold
            if score1 > confidence_threshold and i not in ignore_indices:
                cropped_image = image[box1[1]:box1[3], box1[0]:box1[2]]
                model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
                model_l = ConvolutionalModel(model, output_shape=3, in_features=512)
                model_l.load_state_dict(torch.load(model_path)) #Strike
                model_l.to(device)
                model_l.eval()
                predicted_index = self.classify_image(model_l, cropped_image)
                print(predicted_index)
                
                if predicted_index == 0 or predicted_index == 2:
                    print("Predicted label:", predicted_index)
                    Marked_item+=1

                    # Save cropped images with boxes as new temp images
                    temp_image_path = f"temp_image_{i}.png"
                    cv2.imwrite(temp_image_path, cropped_image)
                    cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)

                    # Update the original image with boxes
                    cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), color, 2)

                    label_position = (box1[0] + 20, box1[3] - 20)
                    if predicted_index == 0:
                        input_tensor = preprocess_image(temp_image_path)
                        pred_name = self.classify_secondary_model(input_tensor=input_tensor,category= 'Animal',v=0)
                        
                    else:
                        input_tensor = preprocess_image(temp_image_path)
                        pred_name = self.classify_secondary_model(input_tensor=input_tensor,category= 'Plant',v=2)
                    
                    di[Marked_item]=pred_name
                    cv2.putText(image, str(Marked_item), label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display the temp image
                    # temp_image = cv2.imread(temp_image_path)
                    # cv2.imshow(f'Temp Image {i}', temp_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    os.remove(temp_image_path)

        if Marked_item==0:
            self.name="No Vaild Object Detected!"
        else:
            print(f"Marked Images = {Marked_item}")
        image=cv2.resize(image, (512,512), interpolation=cv2.INTER_AREA)
        output_path = f"Detected{self.i}.png"
        cv2.imwrite(output_path, image)
        self.roi_im = output_path
        os.remove(output_path)
        self.i = self.i + 1
        return di
    
    def is_box_contained(self,larger_box, smaller_box):
        """
        Check if the larger bounding box completely contains the smaller one.
        """
        return (larger_box[0] <= smaller_box[0] and
                larger_box[1] <= smaller_box[1] and
                larger_box[2] >= smaller_box[2] and
                larger_box[3] >= smaller_box[3])
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def get_name(self):
        image_path = r'temp_roi.jpg' 
        if os.path.exists(image_path):
            input_tensor = preprocess_image(image_path)

            predicted_index = predict_image(input_tensor)

            if predicted_index == 0:
                self.name=""
                file_path = os.path.join(current_directory, r'Model\animalsecondlayer.txt')# animal list path
                with open(file_path, 'r') as file:
                    content = file.read()
                    data_list = ast.literal_eval(content)
                    print(data_list)
                predictedsecond=predict_second_layer(Name="Second Layer Model Called For Animal",input_tensor=input_tensor,output_shape=len(data_list),file_path=os.path.join(current_directory, r'Model\animalsecondlayer.pth')) # animal model path
                S2=data_list[predictedsecond]
                print(S2)
                self.name = "Animal " + " / " + str(S2)
                nam=S2
                fol = os.path.join(current_directory, 'Animals')
                destination_path = os.path.join(fol, f"{S2}.png")
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                shutil.copyfile(image_path, destination_path)
                shutil.copyfile(image_path, f"Animal{self.i}.png")
                self.roi_im = f"Animal{self.i}.png"
                os.remove(f"Animal{self.i}.png")
                self.i = self.i+1
                n=1
                try:
                    while n:
                        self.search()
                        if(self.info==""):
                            n=1
                        else:
                            n=0
                except Exception as e:
                    self.show_popup2()
                    self.info = "No Information Found Try our chat bot by Clicking on below button or Check your internet connection."

            elif predicted_index == 2:
                self.name="Plant"
                file_path = os.path.join(current_directory, r'Model\plantsecondlayer.txt') # plant list path
                with open(file_path, 'r') as file:
                    content = file.read()
                    data_list = ast.literal_eval(content)
                    print(data_list)
                predictedsecond=predict_second_layer(Name="Second Layer Model Called For Plant",input_tensor=input_tensor,output_shape=len(data_list),file_path=os.path.join(current_directory, r'Model\secondplantfinal.pth')) # plant model path
                S2=data_list[predictedsecond]
                print(S2)
                nam = S2
                self.name+=" / "+S2
                fol = os.path.join(current_directory, 'Plants')
                destination_path = os.path.join(fol, f"{S2}.png")
                if os.path.exists(destination_path):
                    os.remove(destination_path)
                shutil.copyfile(image_path, destination_path)
                shutil.copyfile(image_path, f"Plant{self.i}.png")
                self.roi_im = f"Plant{self.i}.png"
                os.remove(f"Plant{self.i}.png")
                self.i = self.i+1
                n=1
                try:
                    while n:
                        self.search()
                        if(self.info==""):
                            n=1
                        else:
                            n=0
                except Exception as e:
                    self.show_popup2()
                    self.info = "No Information Found Try our chat bot by Clicking on below button or Check your internet connection."
            else:
                self.name="Neither Plant Nor Animal Detected"
                S2 = f"Other Detected{self.i}"
                shutil.copyfile(image_path,f"{S2}.png")
                self.roi_im = f"{S2}.png"
                self.info = "Nothing to Search for Please Select vaild Image or vaild ROI to detect Plant and Animals only."
                os.remove(f"{S2}.png")
                self.i = self.i+1
                
        else:
            self.show_popup1()
    
    def bot_link(self):
        webview.create_window("Nature Spy Bot","https://naturespybot.netlify.app/")
        webview.start()

    def show_popup(self):
        title = "Image Not selected"
        message = "Please Select a Image First !!"
        dialog = MDDialog(
            title=title,
            type="alert",
            text=message,
            size_hint=(0.8, 0.3),
            auto_dismiss=True,
            buttons=[
                MDFillRoundFlatIconButton(
                    text="Close",
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()

    def show_popup1(self):
        title = "Roi not select"
        message = "Please select Roi"
        dialog = MDDialog(
            title=title,
            type="alert",
            text=message,
            size_hint=(0.8, 0.3),
            auto_dismiss=True,
            buttons=[
                MDFillRoundFlatIconButton(
                    text="Close",
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()

    def show_popup2(self):
        title = "No information Found"
        message = "Please Try our Chat bot for information"
        dialog = MDDialog(
            title=title,
            type="alert",
            text=message,
            size_hint=(0.8, 0.3),
            auto_dismiss=True,
            buttons=[
                MDFillRoundFlatIconButton(
                    text="Close",
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()


class ImageGrid(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 3
        self.spacing = [10, 10]
        self.padding = [10, 10]
        self.size_hint_y = None
        self.bind(minimum_height=self.setter('height'))
        self.load_images()

        Clock.schedule_interval(self.reload_images, 10)

    def load_images(self):
        fol = os.path.join(current_directory, 'Plants')
        if not os.path.exists(fol):
            os.makedirs(fol)
        folder_path = fol
        self.clear_widgets() 

        files = os.listdir(folder_path)

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                box = BoxLayout(orientation='vertical', size_hint_y=None, height=200)
                img = i(source=os.path.join(folder_path, file), allow_stretch=True)
                lbl = Label(text=os.path.splitext(file)[0], size_hint_y=None, height=50,color='black',bold=True)
                box.add_widget(img)
                box.add_widget(lbl)
                self.add_widget(box)

    def reload_images(self,dt):
        self.load_images()


class ImageGrid2(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cols = 3
        self.spacing = [10, 10]
        self.padding = [10, 10]
        self.size_hint_y = None
        self.bind(minimum_height=self.setter('height'))
        self.load_images()

        Clock.schedule_interval(self.reload_images, 10)

    def load_images(self):
        fol = os.path.join(current_directory, 'Animals')
        if not os.path.exists(fol):
            os.makedirs(fol)
        folder_path = fol
        self.clear_widgets() 

        files = os.listdir(folder_path)

        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                box = BoxLayout(orientation='vertical', size_hint_y=None, height=200)
                img = i(source=os.path.join(folder_path, file), allow_stretch=True, size_hint_y=None)
                lbl = Label(text=os.path.splitext(file)[0], size_hint_y=None, height=50,color='black',bold=True)
                box.add_widget(img)
                box.add_widget(lbl)
                self.add_widget(box)

    def reload_images(self,dt):
        self.load_images()

if __name__ == '__main__':
    NatureEyes().run()
    MainApp().run()
