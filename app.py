# importing all the libraries and dependancies that are required.
from flask import Flask, render_template, request, redirect, url_for, send_file


from datetime import date
from datetime import datetime
import os
import cv2
import re
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np
import pandas as pd
#  Scikit-learn provides a wide range of tools for various machine learning tasks
#  such as classification, regression, and clustering. In the context of face recognition,
#  it can be used for training and evaluating classification models that can identify individuals from their facial features.


# Classification: Classification is a supervised learning task where the goal is to predict the category or class of a new input based on its features.
# declaring the flask app name.

# Joblib, on the other hand, is a library that provides tools for parallel computing and caching in Python.
# It is often used in conjunction with scikit-learn to speed up the training and deployment of machine learning models.


app = Flask(__name__)


# Using a dictionary to store the username and password.
users = {}


# The users will be directed to this page initally when we click on the link.
@app.route('/')
def index():
    return render_template('login.html')  #rendering the login template first.


# The route for login,this function will be called when we click on Login Button.
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username'] # Collecting the username from the Frontend.
    password = request.form['password'] # Collecting the Password from the Frontend.
    if username in users and users[username] == password: # Checking if the useranme is there in the users dictionary.
        return redirect(url_for('main')) # If it is then it should be redirected to the main page.
    else:
        return render_template('login.html', message='Invalid username or password')# If not then display the Invalid message.


# The Register Route.
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET': # If the method is get then render the Register page
        return render_template('register.html')
    else:
        username = request.form['username'] # Collecting the username from the Frontend.
        password = request.form['password'] # Collecting the Password from the Frontend.
        confirm = request.form['confirm'] # Collecting the Confirm Password from the Frontend.
        email = request.form['email'] # Collecting the Confirm Password from the Frontend.
        password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'# Regular Expression for Password.
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'# Regular Expression for Email.

        # Confirming Validations with the information that is taken.
        if username in users:
            return render_template('register.html', message='Username already taken')# Checking if the Username is already Taken.
        elif not re.match(email_regex, email):# Checking the Email Regex.
            return render_template('register.html', message='Invalid email address.')# Email regex validation
        elif not re.match(password_regex, password):# Password rules: At least 8 characters, one uppercase letter, one lowercase letter, one digit, and one special character
            return render_template('register.html', message='Invalid password. Password should have at least 8 characters, one uppercase letter, one lowercase letter, one digit, and one special character.')
        elif password != confirm:
            return render_template('register.html', message='Passwords do not match')# Checking if the Password and the Confirrm Passwords Match.
        else:
            # If the Validations are Passed then Add the new user to the dictionary of users.
            users[username] = password
            return redirect(url_for('index')) # Give the url for Login Once Registering.


# From Here The main functions and Parameters Required for the Attendance System are declared from Here.


# Getting the current date .
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


# Creating  a Directory to store the Attendance Sheet.
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')# If the Attendance Directory is not there then create One.


# Creating a Directory to store the Faces of Students captured by webcam.
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')# If the Static Directory is not there then create One.


# Creating a CSV file with Today's Date to store the Attendance of Student's
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')#creates a face detector object
# the haarcascade_frontalface_default.xml file contains pre-trained classifiers for detecting faces in images or videos.
cap = cv2.VideoCapture(0)
# initializes a video capture object that captures video from the default camera (i.e., the camera with index 0).
# This object is used to capture the video feed from the camera and process it in real-time.


# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#converts the input image from the default BGR color space to grayscale.
    face_points = face_detector.detectMultiScale(gray, 1.3, 5) #applies the face detector object (face_detector) created earlier to the grayscale image (gray) to detect any faces present in the image.
    # The detectMultiScale method uses a sliding window approach to scan the image at multiple scales and detect objects of different sizes.
    # The function returns a list of bounding boxes, where each bounding box specifies the coordinates (x, y, width, height) of a detected face region.
    return face_points


# The main function where the ML model will give the output
# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#Trainning the Model Here Only.
def train_model():
    # Initializing empty lists faces and labels to store the face images and corresponding labels.
    faces = []
    labels = []
    #list of all subdirectories under the static/faces directory, each of which corresponds to a different person's face.
    userlist = os.listdir('static/faces')
    for user in userlist:  #For each subdirectory, loop through all the image files inside it.
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}') #Read each image file using the OpenCV library (cv2.imread).
            resized_face = cv2.resize(img, (50, 50)) #Resize each image to a fixed size of 50x50 pixels using the OpenCV cv2.resize function.
            faces.append(resized_face.ravel()) #Flatten the resized image into a 1D NumPy array using the ravel() method.
            labels.append(user) # Append the flattened image to the faces list and the corresponding label (the name of the subdirectory/person) to the labels list.
    faces = np.array(faces) #Convert the faces list to a NumPy array.
    knn = KNeighborsClassifier(n_neighbors=5) #Create a new instance of the KNeighborsClassifier class with n_neighbors=5
    # (the number of nearest neighbors to consider for classification).
    knn.fit(faces,labels)
    # Train the model using the fit method, passing in the faces array as the training data and the labels list as the corresponding labels.
    joblib.dump(knn,'static/face_recognition_model.pkl')
    # Save the trained model to a file called face_recognition_model.pkl using the joblib.dump method from the joblib library.


# Extracting today's Attendance from the csv file.
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names,rolls,times,l


# Adding the Attendabce of the student whoes name is passed as the parameter
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')



# This is the Main Page of our Website.
@app.route('/main')
def main():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    while ret:
        ret,frame = cap.read()
        if extract_faces(frame)!=():
            (x,y,w,h) = extract_faces(frame)[0]
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h,x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1,-1))[0]
            print(identified_person)
            add_attendance(identified_person)
            cv2.putText(frame,f'{identified_person}',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
        cv2.imshow('Attendance',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,l = extract_attendance()
    return render_template('home.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)



@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,l = extract_attendance()
    return render_template('home1.html',names=names,rolls=rolls,times=times,l=l,totalreg=totalreg(),datetoday2=datetoday2)

@app.route('/h1',methods=['GET'])
def h1():

    return render_template('home1.html',totalreg=totalreg())


@app.route('/download_attendance')
def download_attendance():
    filename = f'Attendance/Attendance-{datetoday}.csv'
    return send_file(filename, as_attachment=True)
if __name__ == '__main__':
    app.run()


