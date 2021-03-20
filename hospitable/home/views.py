from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.contrib.auth import login, logout, authenticate
from .Forms import RecordForm
from django.http.response import StreamingHttpResponse
from .models import Record
from home.camera import VideoCamera
from django.contrib.auth.decorators import login_required
from django.template import loader, Context

camera = VideoCamera()

def gen():
	global camera
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
	return StreamingHttpResponse(gen(),
					content_type='multipart/x-mixed-replace; boundary=frame')


def home(request):
    return render(request, 'home.html')

def dumbmode(request):
    return render(request, 'dumb.html')

def blindmode(request):
    return render(request, 'blind.html')

def deafmode(request):
    return render(request, 'deaf.html')

@login_required
def profile(request):
    return render(request, 'profile.html')

def signup(request):
    logout(request)
    if request.method == 'GET':
        return render(request, 'signup.html', {'form': UserCreationForm()})
    else:
        if request.POST['password1'] == request.POST['password2']:
            try:
                email = request.POST['email'].lower()
                r = User.objects.filter(email=email)
                if r.count():
                    return render(request, 'signup.html', {'error': 'Email already exists'})
                else:
                    user = User.objects.create_user(request.POST['username'], password=request.POST['password1'], email=request.POST['email'])
                    user.save()
                    login(request, user)
                    return redirect('home')

            except IntegrityError:
                return render(request, 'signup.html',{'error': 'This username has already been taken. Please choose a new Username'})
        else:
            return render(request, 'signup.html', {'error': 'Passwords did not match'})


def loginuser(request):
    logout(request)
    if request.method == 'GET':
        return render(request, 'loginuser.html', {'form': AuthenticationForm()})
    else:
        user = authenticate(request, username=request.POST['username'], password=request.POST['password'])
        if user is None:
            return render(request, 'loginuser.html',
                          {'form': AuthenticationForm(), 'error': 'User password did not match'})
        else:
            login(request, user)
            return redirect('home')

@login_required
def logoutuser(request):
    if request.method == 'POST':
        logout(request)
        return redirect('home')

@login_required
def uploadrecord(request):
    if request.method == 'GET':
        return render(request, 'recordupload.html', {'form': RecordForm()})
    else:
        try:
            form = RecordForm(request.POST, request.FILES)
            newRecord = form.save(commit=False)
            newRecord.user = request.user
            newRecord.save()
            return redirect('listofrecords')
        except ValueError:
            return render(request, 'recordupload.html',
                          {'form': RecordForm(), 'error': 'Wrong data put in. Try Again'})

@login_required
def listofrecords(request):
    record = Record.objects.filter(user=request.user).order_by('-created')
    return render(request, 'recordlist.html', {'record':record})