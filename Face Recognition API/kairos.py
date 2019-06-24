import requests
import numpy as np
from urllib.request import urlopen
import cv2
import json

class kairos:
    def __init__(self):
        self.url = "https://kairosapi-karios-v1.p.rapidapi.com/"
        self.headers = {
            "X-RapidAPI-Host": "kairosapi-karios-v1.p.rapidapi.com",
            "X-RapidAPI-Key": "6f5954cbc7msh8b7b67ccd6ccb4ep119c51jsn55cfaa4c822b",
            "Content-Type": "application/json"
        }

    def ListAllGalleries(self):
        action = "gallery/list_all"
        url = self.url + action
        response = requests.request('POST', url, headers=self.headers)
        return response.json()

    def GalleryViewSubject(self, gallery_name, subject_id):
        action = "gallery/view_subject"
        url = self.url + action
        params = json.dumps({u"gallery_name": gallery_name,
                             u"subject_id": subject_id})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def GalleryView(self, gallery_name):
        action = "gallery/view"
        url = self.url + action
        params = json.dumps({u"gallery_name": gallery_name})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def RemoveGalleries(self, gallery_name):
        action = "gallery/remove"
        url = self.url + action
        params = json.dumps({u"gallery_name": gallery_name})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def GalleriesRemoveSubject(self, gallery_name, subject_id):
        action = "gallery/remove_subject"
        url = self.url + action
        params = json.dumps({u"gallery_name": gallery_name,
                             u"subject_id": subject_id})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def Enroll(self, image_url, gallery_name, subject_id):
        action = "enroll"
        url = self.url + action
        params = json.dumps({u"image": image_url,
                             u"gallery_name": gallery_name,
                             u"subject_id": subject_id})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def Verify(self, image, gallery_name, subject_id):
        action = "verify"
        url = self.url + action
        params = json.dumps({u"image": image,
                             u"gallery_name": gallery_name,
                             u"subject_id": subject_id})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def Recognize(self, image, gallery_name):
        action = "recognize"
        url = self.url + action
        params = json.dumps({u"image": image,
                             u"gallery_name": gallery_name})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

    def Detect(self, image):
        action = "detect"
        url = self.url + action
        params = json.dumps({u"image": image,
                             u"selector": u"ROLL"})
        response = requests.request('POST', url, headers=self.headers, data=params)
        return response.json()

        """topLeftX = response['images'][0]['faces'][0]['topLeftX']
        topLeftY = response['images'][0]['faces'][0]['topLeftY']
        height = response['images'][0]['faces'][0]['height']
        width = response['images'][0]['faces'][0]['width']

        resp = urlopen(image)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)


        cv2.rectangle(image, (topLeftX, topLeftY), (topLeftX + width, topLeftY + height),
                     (0, 255, 0), 2)
        cv2.imshow('face', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
