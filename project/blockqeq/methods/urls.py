# coding=utf-8
from django.conf.urls import include
from django.urls import path

from methods.views import hello, abscheme, naive, shamir, blakley, abscheme_recover, mignotte_recover, \
    naive_recover, shamir_recover, blakley_recover, mignotte_sequence, mignotte_second_step

urlpatterns = {
    path(r'', hello),
    path(r'ab/', abscheme),
    path(r'mignotte/', mignotte_sequence),
    path(r'mignotte_second_step/', mignotte_second_step),
    path(r'naive/', naive),
    path(r'shamir/', shamir),
    path(r'blakley/', blakley),
    path(r'ab_recover/', abscheme_recover),
    path(r'mignotte_recover/', mignotte_recover),
    path(r'naive_recover/', naive_recover),
    path(r'shamir_recover/', shamir_recover),
    path(r'blakley_recover/', blakley_recover),

    # path(r'about/', about),
    # path(r'surveys/<int:id>/', SurveyDetail.as_view(), name='survey-detail-my'),
    # path(r'result/<int:sid>/<str:uuid>/', predict, name='predict-gift')
}