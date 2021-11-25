from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
# Create your views here.
from django.template import loader
from hand_tracker.app import main


def index(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render({}, request))


def assets(request):
    template = loader.get_template('index.html')
    return HttpResponse(template.render({}, request))


def video_feed(request):
    return StreamingHttpResponse(main(), content_type='multipart/x-mixed-replace; boundary=frame')
