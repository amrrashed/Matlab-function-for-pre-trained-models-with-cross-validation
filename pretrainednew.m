clc;clear all;close all
%x=cell(4,5);
[t,x]=resnet18cv('G:\new researches\mansour paper\crop224new','adam',0,5);
t=ceil(t.*10000)/10000;
pause(60)
[t1,x1]=resnet18cv('G:\new researches\mansour paper\crop224new','rmsprop',0,5);
t1=ceil(t1.*10000)/10000;
pause(60)
[t2,x2]=resnet18cv('G:\new researches\mansour paper\dataset224','adam',0,5);
t2=ceil(t2.*10000)/10000;
pause(60)
[t3,x3]=resnet18cv('G:\new researches\mansour paper\dataset224','rmsprop',0,5);
t3=ceil(t3.*10000)/10000;
pause(60)
title={'''AUC','''accuracy','''sensitivity','''specificity','''precision','''recall','''f_measure','''gmean'};
total=[t;t1;t2;t3];
filename='performance.xlsx';
xlswrite(filename,title,'Sheet1','A1')
xlswrite(filename,total,'Sheet1','A2')
winopen(filename);
