clc;
clear;
close all;
clear variables;

%% Wichtigste Parameter
% Wichtig: innerhalb einer Simulationskette nach Möglichkeit nur eine
% Abtastrate verwenden!!!!!
fs=81920;           % Abtastrate 
f0=800;             % Grundfrequenz 
f1=900;             % Grundfrequenz 
f2=1600;            % Grundfrequenz 
f3=2000;            % Grundfrequenz 
f4=3200;            % Grundfrequenz 



simdur=1;           % Zeitspanne, die simuliert werden soll (1s)
ts=1/fs;            % Abtastperiode
Nsamps=simdur/ts;   % Anzahl Abtastwerte, die erzeugt werden

%% Sinus-Signale
sig0=create_sin_rect(fs,f0,-1,Nsamps);
sig1=create_sin_rect(fs,f1,-1,Nsamps);
sig2=create_sin_rect(fs,f2,-1,Nsamps);
sig3=create_sin_rect(fs,f3,-1,Nsamps);

% 'leere' Signalstruktur als Vorlage für andere Signale erzeugen
sig_template.tsample=sig0.tsample;
sig_template.tstart=sig0.tstart;

% zwei 'Rauschsignale' erzeugen
sig4=sig_template;
sig4.samples=randn(size(sig0.samples));
sig5=sig_template;
sig5.samples=randn(size(sig0.samples));

%%----------------------------------------------------------------------------------------------------------------
%%Aufgabe1
%%Allgemeine vorgabe T=10ms
T=0.01;

%%signal1
%Signalspezifische Variablen -->n muss nur in plus und minus interschieden werden!
Tp=T; %10ms Periodendauer
Ts=20*1e-6; %Abtastfrequenz

%Ans für alle Fälle
A_plus_signal1=0.5;
A_minus_signal1=0.5;

%Zeitvektor
tvec = -50/1000: Ts : 50/1000;
vec=0*tvec; %ver tvec zuweisen = 0

%Summenfunktion
vec=vec+A_plus_signal1*exp(1i*2*pi*1/Tp*tvec);
vec=vec+A_minus_signal1*exp(1i*2*pi*1/Tp*-1*tvec);   

%Ergebnis FourierSynthese plotten
figure();
plot(tvec, vec);
dodo=axis;
dodo(1)=-0.05; dodo(2)=0.05;
dodo(3)=-1.1; dodo(4)=1.1;
axis(dodo);
title('sig1(t) nach Fourier-Synthese');
xlabel('t in s');
ylabel('Amplitude');

%Frequenzspektrum plotten
vec1.samples = vec;
vec1.tsample= Ts;
plot_signal_spectrum(vec1);

%OrginalSignal
sig1 = cos(2*pi*tvec/T);
plot(tvec,sig1);
title('sig1(t) OrginalSignal');
xlabel('t in s');
ylabel('Amplitude');

%-------------------------------------------------------------------------------------------------
%%signal2
%Signalspezifische Variablen -->n muss nur in plus und minus interschieden werden!
Tp=3*T; %Periodendauer 30ms
Ts=20*1e-6; %Abtastfrequenz

%Ans für alle Fälle
A_plus_signal2=0.5;
A_minus_signal2=0.5;

%Zeitvektor
tvec = -50/1000: Ts : 50/1000;
vec=0*tvec; %ver tvec zuweisen = 0

%Summenfunktion
vec=vec+A_plus_signal2*exp(1i*2*pi*1/Tp*tvec);
vec=vec+A_minus_signal2*exp(1i*2*pi*1/Tp*-1*tvec);   

%Ergebnis FourierSynthese plotten
figure();
plot(tvec, vec);
dodo=axis;
dodo(1)=-0.05; dodo(2)=0.05;
dodo(3)=-1.1; dodo(4)=1.1;
axis(dodo);
title('sig2(t) nach Fourier-Synthese');
xlabel('t in s');
ylabel('Amplitude');

%Frequenzspektrum plotten
vec2.samples = vec;
vec2.tsample= Ts;
plot_signal_spectrum(vec2);

%OrginalSignal
sig2 = cos(2*pi*tvec/(3*T));
plot(tvec,sig2);
title('sig2(t) Frequenzspektrum');
xlabel('t in s');
ylabel('Amplitude');

%-------------------------------------------------------------------------------------------------
%%signal3
%Signalspezifische Variablen
Tp=20/1000; 
Ts=20*1e-6; 

%verschiedene ns
n=50; %n je nach graph anpassen
nvec=-n:1:n;
nvec_eve_signal3=-n:2:n;
nvec_odd_signal3=-n+1:2:n-1;

%Ans für alle Fälle
An_null_signal3=0.5;
An_even_signal3=0;
An_odd_signal3=(1./(pi.*nvec_odd_signal3)).*((-1).^((nvec_odd_signal3-1)/2));

%Zeitvektor
tvec = -50/1000: Ts : 50/1000;
vec=0*tvec;

%Summenfunktion
for it=1:length(nvec_odd_signal3)
    n=nvec_odd_signal3(it);
    An=An_odd_signal3(it);
    vec=vec+An*exp(1i*2*pi*f0*n*tvec);       
end
%An_null nicht vergessen
vec=vec+An_null_signal3;

%Ergebnis FourierSynthese plotten
figure();
plot(2*tvec/Tp, vec);
title('sig3(t) nach Fourier-Synthese');
dodo=axis;
dodo(1)=-0.05; dodo(2)=0.05;
dodo(3)=-0.1; dodo(4)=1.1;
axis(dodo);
xlabel('t in s');
ylabel('Amplitude');

%Frequenzspektrum plotten
vec3.samples = vec;
vec3.tsample= Ts;
plot_signal_spectrum(vec3);

%OrginalSignal
figure();
t=-4*pi:ts*100:4*pi;
sig3=0.5*square(t+0.5*pi)+0.5;
plot(t/pi,sig3);
title('sig3 OrignalFunktion');
dodo=axis;
dodo(1)=-4; dodo(2)=4;
dodo(3)=-0.1; dodo(4)=1.1;
axis(dodo);
xlabel('t in T');
ylabel('Amplitude');

%-------------------------------------------------------------------------------------------------
%%signal4
%Signalspezifische Variablen
Tp=20/1000; 
Ts=20*1e-6; 

%verschiedene ns
n=50; %n je nach graph anpassen
nvec=-n:1:n;
nvec_eve_signal4=-n:2:n;
nvec_odd_signal4=-n+1:2:n-1;

%Ans für alle Fälle
An_null_signal4=0.5;
An_even_signal4=0;
An_odd_signal4=-(1i./(pi.*nvec_odd_signal4));

%Zeitvektor
tvec = -50/1000: Ts : 50/1000;
vec=0*tvec;

%Summenfunktion
for it=1:length(nvec_odd_signal4)
    n=nvec_odd_signal4(it);
    An=An_odd_signal4(it);
    vec=vec+An*exp(1i*2*pi*f0*n*tvec);       
end
%An_null nicht vergessen
vec=vec+An_null_signal4;

%Ergebnis FourierSynthese plotten
figure();
plot(2*tvec/Tp, vec);
title('sig4(t) nach Fourier-Synthese');
dodo=axis;
dodo(1)=-0.05; dodo(2)=0.15;
dodo(3)=-0.1; dodo(4)=1.1;
axis(dodo);
xlabel('t in s');
ylabel('Amplitude');

%Frequenzspektrum plotten
vec4.samples = vec;
vec4.tsample= Ts;
plot_signal_spectrum(vec4);

%OrginalSignal
figure();
t=-4*pi:ts*100:4*pi;
sig4=0.5*square(t)+0.5;
plot(t/pi,sig4);
title('sig4 OrignalFunktion');
dodo=axis;
dodo(1)=-4; dodo(2)=4;
dodo(3)=-0.1; dodo(4)=1.1;
axis(dodo);
xlabel('t in T');
ylabel('Amplitude');

%-------------------------------------------------------------------------------------------------
%%Aufgabe2
%Die beiden Rechtecksignale bereits in Aufgabe 1 angelegt:
%vec 3 spektrum
vec3_2.samples = sig3;
vec3_2.tsample= Ts;
plot_signal_spectrum(vec3_2);
%vec4 spektrum
vec4_2.samples = sig4;
vec4_2.tsample= Ts;
plot_signal_spectrum(vec4_2);


%Sinus-Signal mit Frequenz 500Hz wird erzeugt.
signal2b=create_sin_rect(fs,500,-1,Nsamps);

%Signal 2b Zeitsignal wird geplottet
plot_timesig(signal2b);
title('signal 2b)');
dodo=axis;
dodo(1)=0; dodo(2)=0.01;
dodo(3)=-1.1; dodo(4)=1.1;
axis(dodo);
%Signalspektrum:
plot_signal_spectrum(signal2b);

%gefenstertes Sinus-Signal mit Frequenz 500Hz wird erzeugt.
signal2c=create_sin_rect(fs,500,20,Nsamps);
%Signal 2c Zeitsignal wird geplottet
plot_timesig(signal2c);
title('signal 2c)');
dodo=axis;
dodo(1)=-0.03; dodo(2)=0.03;
dodo(3)=-1.1; dodo(4)=1.1;
axis(dodo);
%Signalspektrum:
plot_signal_spectrum(signal2c);

%%d) sinc signal erzeugen und plotten
signal2d=create_sinc_rect(fs,125,-1,Nsamps);
plot_timesig(signal2d)
title('signal 2d)')
dodo=axis;
dodo(1)=-0.5; dodo(2)=0.5;
axis(dodo);
%Signalspektrum:
plot_signal_spectrum(signal2d);

%-------------------------------------------------------------------------------
%% Funktionen 

%% Korrelationsfunktion entsprechend der Definition aus der Vorlesung
function phi_xy=xcorr_RT(x,y)
phi_xy=xcorr(x,y);
phi_xy=phi_xy(end:-1:1);  % Vektor in umgekehrter Reihenfolge anordnen
return
end

%% Erzeugung eines cosinus signals
% fs sampling rate
% f0 base frequency of cos
% N: number of cosinusoidal periods within rect; if -1: no windowing
function sig_out=create_cos_rect(fs,f0,N,Nsamples)
tsample=1/fs;
% create a timevector with Nsamples/2 samples in negative time region
% and Nsamples/2 in positiv time region
timevec=-(Nsamples-1)/2*tsample:tsample:Nsamples/2*tsample;
% duration of rectangular pulse in order to fit with N perios
% of the sinusoidal
if N<0
    rectdur=2*Nsamples;
else
    rectdur=1/f0*N;
end

% rectangular samples
rect_samps=rectpuls(timevec/rectdur);
% cosinusoidal samples
cos_samples=cos(2*pi*f0*timevec);
% multiply rectangular signal with sinusoidal signal
sig_out.samples=rect_samps.*cos_samples;
sig_out.tsample=tsample;
% as we deal here with negative time values, and not with time starting
% at t=0, it is useful to store the starting time of the time axis
sig_out.tstart=timevec(1);
return;
end

%% Erzeugung eines sinus signals
% fs sampling rate
% f0 base frequency of sin
% N: number of soinusoidal periods within rect; if -1: no windowing
function sig_out=create_sin_rect(fs,f0,N,Nsamples)
tsample=1/fs;
% create a timevector with Nsamples/2 samples in negative time region
% and Nsamples/2 in positiv time region
timevec=-(Nsamples-1)/2*tsample:tsample:Nsamples/2*tsample;
% duration of rectangular pulse in order to fit with N perios
% of the sinusoidal
if N<0
    rectdur=2*Nsamples;
else
    rectdur=1/f0*N;
end

% rectangular samples
rect_samps=rectpuls(timevec/rectdur);
% sinusoidal samples
sin_samples=sin(2*pi*f0*timevec);
% multiply rectangular signal with sinusoidal signal
sig_out.samples=rect_samps.*sin_samples;
sig_out.tsample=tsample;
% as we deal here with negative time values, and not with time starting
% at t=0, it is useful to store the starting time of the time axis
sig_out.tstart=timevec(1);
return;
end
%% Erzeugung eines sinc signals
% fs sampling rate
% f0 base frequency of sinc 
% N: number of 'sinc-swings' within rect; if -1: no windowing
function sig_out=create_sinc_rect(fs,f0,N,Nsamples)
tsample=1/fs;
% create a timevector with Nsamples/2 samples in negative time region
% and Nsamples/2 in positiv time region
timevec=-(Nsamples-1)/2*tsample:tsample:Nsamples/2*tsample;
% duration of rectangular pulse in order to fit with N periods
% of the sinusoidal
if N<0
    rectdur=2*Nsamples;
else
    rectdur=1/f0*N;
end

% rectangular samples
rect_samps=rectpuls(timevec/rectdur);
% sinusoidal samples
sin_samples=sinc(f0*timevec);
% multiply rectangular signal with sinusoidal signal
sig_out.samples=rect_samps.*sin_samples;
sig_out.tsample=tsample;
% as we deal here with negative time values, and not with time starting
% at t=0, it is useful to store the starting time of the time axis
sig_out.tstart=timevec(1);
return;
end
%% Funktion um Zeitsignal darzustellen
function plot_timesig(sig)
% extract the components from the structure. This is not necessary, but
% simplifies the later code as you can work with shorter terms 
% e.g. with ts instead of sig.tsample

% create the time axis that starts at sig.tstart
len=length(sig.samples);
tsample=sig.tsample;
tstart=sig.tstart;
timevec=tstart: tsample : tstart+(len-1)*tsample;

% open a new figure
figure();
if imag(sig.samples)==0
    plot(timevec, sig.samples);
else
    plot(timevec, real(sig.samples),'b+',timevec, imag(sig.samples),'g*');
    legend('real','imag');
end

% annotate the plot
xlabel('time [s]');
ylabel('amplitude [V]');
title('timesignal');
% add a grid to the plot
grid 'on';
return;
end
%% Funktion um Signalspektrum darzustellen
function [fvec,spec]=plot_signal_spectrum(insig)
% get the parameters
samples = insig.samples;    % sample values
sr= 1/insig.tsample;        % sampling rate
fstart=-sr/2;               % lowest frequency in the plot
fstop=  sr/2;               % highest calculated frequency
n=length(samples);          % number of samples

% dft mit windowing
spec=fftshift(fft(samples.*(hanning(n)).')/(n)); % spectrum

% dft ohne windowing
% spec=fftshift(fft(samples)/(n)); % spectrum
powspec=spec.*conj(spec); % power
% power in dBm (assuming that the signal is in [V] 
% and the measurement resistor is 1 Ohm
powdbm=10*log10(powspec*1000); 
fvec=linspace(fstart,fstop,n); % create the frequency vector

% create the plot
figure();
plot(fvec,powdbm,'m', 'LineWidth', 2);
% legend('real')
axis tight;
ylabel('[dBm]');
xlabel('frequency [Hz]');
title('power spectrum [dBm]');
grid 'on';
grid 'minor';
return;
end