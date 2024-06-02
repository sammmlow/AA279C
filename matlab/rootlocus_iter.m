clc; close all; clear all;
s = tf('s');

Ix = 4770.398;
Iy = 6313.894;
Iz = 7413.202;

%% Get estimate of Kp
Kdx = -1;
Kdy = -1;
Kdz = -1;
sys1x = 1 / (Ix * s^2 - Kdx * s);
sys1y = 1 / (Iy * s^2 - Kdy * s);
sys1z = 1 / (Iz * s^2 - Kdz * s);
figure(1); rlocusplot(sys1x); grid on;
figure(2); rlocusplot(sys1y); grid on;
figure(3); rlocusplot(sys1z); grid on;

%% Get estimate of Kd
Kpx = -5.24E-5;
Kpy = -3.99E-5;
Kpz = -3.37E-5;
sys2x = s / (Ix * s^2 - Kpx);
sys2y = s / (Iy * s^2 - Kpy);
sys2z = s / (Iz * s^2 - Kpz);
figure(4); rlocusplot(sys2x); grid on;
figure(5); rlocusplot(sys2y); grid on;
figure(6); rlocusplot(sys2z); grid on;