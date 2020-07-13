clc;
clear;
close all;
clear variables;

%% Wichtigste Parameter
% Wichtig: innerhalb einer Simulationskette nach Möglichkeit nur eine
% 'leere' Signalstruktur als Vorlage für andere Signale erzeugen

%% Aufgabe1
%%----------------------------------------------------------------------------------------------------------------

%%Allgemeine vorgabe T=10ms
fs = 50000;             %Abtastrate
Ts = 1/fs;              %Abtastperiode
tvec = -0.5: Ts : 0.5;  %Zeitvektor
tvec = tvec(1:50000);   %Zeitvektor Laenge anpassen

simdur=1;           % Zeitspanne, die simuliert werden soll (1s)
ts=1/fs;            % Abtastperiode
Nsamps=simdur/ts;   % Anzahl Abtastwerte, die erzeugt werden

%%Signale erzeugen
%signala
siga=square(2*pi*500*tvec);

figure();
plot(tvec,siga);
dodo=axis;
dodo(1)=-0.01; dodo(2)=0.01;
dodo(3)=-1.1; dodo(4)=1.1;
axis(dodo);
title('Signal_a alternierende Rechteckfolge 1ms');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

sig_a.samples = siga;
sig_a.tsample= Ts;

%Erzeugung SpektrumPlot
plot_signal_spectrum(sig_a, 'Spektrum S_a  [dBm]');

%%signalb
sigb= create_cos_rect(fs,500,-1,Nsamps);
sig_template.tsample=sigb.tsample;
sig_template.tstart=sigb.tstart;

figure();
plot(tvec,sigb.samples);
dodo=axis;
dodo(1)=-0.01; dodo(2)=0.01;
dodo(3)=-1; dodo(4)=1;
axis(dodo);
title('Signal_b cosinus f = 500Hz');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(sigb , 'Spektrum S_b  [dBm]');

%%signalc
sigc= create_cos_rect(fs,2000,-1,Nsamps);

figure();
plot(tvec,sigc.samples);
dodo=axis;
dodo(1)=-0.005; dodo(2)=0.005;
dodo(3)=-1; dodo(4)=1;
axis(dodo);
title('Signal_c cosinus f = 2000Hz');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(sigc, 'Spektrum S_c  [dBm]');

%%signald
sigd = create_sinc_rect(fs,125,-1,Nsamps);

figure();
plot(tvec,sigd.samples);
dodo=axis;
dodo(1)=-0.2; dodo(2)=0.2;
dodo(3)=-0.25; dodo(4)=1;
axis(dodo);
title('Signal_d sinc 250 Nulldurchgaenge');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(sigd, 'Spektrum S_d  [dBm]' );

%%Systeme erzeugen
%Vektor gefuellt mit 0 erzeugen in richtiger L�nge
nullvec = -0.5: Ts : 0.5;
nullvec = nullvec(1:50000);
nullvec = 0*nullvec;

%%Systema --> dirac mit 1 bei 0 und 1 ms
systema = nullvec;
systema(25001) = 1;
systema(25051) = 1;

figure();
plot(tvec,systema);
dodo=axis;
dodo(1)=-0.01; dodo(2)=0.01;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('System_A Impulsantwort');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');


%%Systemb --> dirac mit 1 bei 0 und 2 ms
systemb = nullvec;
systemb(25001) = 1;
systemb(25101) = 1;

figure();
plot(tvec,systemb);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('System_B Impulsantwort');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%%Systemc --> rect T=2ms
systemc = nullvec;
systemc(24951) = 1;
systemc(25051) = 1;
for it=24951:25051
    systemc(it)=1;       
end

figure();
plot(tvec, systemc);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('System_C Impulsantwort');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');


%%Systemd --> sinc mit T = 2ms
systemd = create_sinc_rect(fs,500,-1,Nsamps);


figure();
plot(tvec, systemd.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=-0.25; dodo(4)=1.1;
axis(dodo);
title('System_D Impulsantwort');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%-------------------------------------------------------------------------------
tvec2 = -0.5: Ts : 1.5;
tvec2 = tvec2(1:99999);
%% System A Faltungen:
%%signala
system_A_a = conv(sig_a.samples, systema);
systemA_a.samples = system_A_a;
systemA_a.tsample= Ts;


figure();
plot(tvec2,systemA_a.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_a mit System_a Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemA_a, 'Faltung Signal_a mit System_A Frequenzbereich');

%%signalb
system_A_b = conv(sigb.samples, systema);
systemA_b.samples = system_A_b;
systemA_b.tsample= Ts;


figure();
plot(tvec2,systemA_b.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_b mit System_a Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemA_b, 'Faltung Signal_b mit System_A Frequenzbereich');

%%signalc
system_A_c = conv(sigc.samples, systema);
systemA_c.samples = system_A_c;
systemA_c.tsample= Ts;


figure();
plot(tvec2,systemA_c.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_c mit System_a Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemA_c, 'Faltung Signal_c mit System_A Frequenzbereich');

%%signalc
system_A_d = conv(sigd.samples, systema);
systemA_d.samples = system_A_d;
systemA_d.tsample= Ts;


figure();
plot(tvec2,systemA_d.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_d mit System_a Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemA_d, 'Faltung Signal_d mit System_A Frequenzbereich');

%% System B Faltungen:
%%signala
system_B_a = conv(sig_a.samples, systemb);
systemB_a.samples = system_B_a;
systemB_a.tsample= Ts;

figure();
plot(tvec2,systemB_a.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_a mit System_B Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemB_a, 'Faltung Signal_a mit System_B Frequenzbereich');

%%signalb
system_B_b = conv(sigb.samples, systemb);
systemB_b.samples = system_B_b;
systemB_b.tsample= Ts;

figure();
plot(tvec2,systemB_b.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_b mit System_B Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemB_b, 'Faltung Signal_b mit System_B Frequenzbereich');

%%signalc
system_B_c = conv(sigc.samples, systemb);
systemB_c.samples = system_B_c;
systemB_c.tsample= Ts;

figure();
plot(tvec2,systemB_c.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_c mit System_B Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemB_c, 'Faltung Signal_c mit System_B Frequenzbereich');

%%signald
system_B_d = conv(sigd.samples, systemb);
systemB_d.samples = system_B_d;
systemB_d.tsample= Ts;

figure();
plot(tvec2,systemB_d.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_d mit System_B Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemB_d, 'Faltung Signal_d mit System_B Frequenzbereich');

%% System C Faltungen:
%%signala
system_C_a = conv(sig_a.samples, systemc);
systemC_a.samples = system_C_a;
systemC_a.tsample= Ts;

figure();
plot(tvec2,systemC_a.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_a mit System_C Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemC_a, 'Faltung Signal_a mit System_C Frequenzbereich');

%%signalb
system_C_b = conv(sigb.samples, systemc);
systemC_b.samples = system_C_b;
systemC_b.tsample= Ts;

figure();
plot(tvec2,systemC_b.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_b mit System_C Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemC_b, 'Faltung Signal_b mit System_C Frequenzbereich');

%%signalc
system_C_c = conv(sigc.samples, systemc);
systemC_c.samples = system_C_c;
systemC_c.tsample= Ts;

figure();
plot(tvec2,systemC_c.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_c mit System_C Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemC_c, 'Faltung Signal_c mit System_C Frequenzbereich');

%%signald
system_C_d = conv(sigd.samples, systemc);
systemC_d.samples = system_C_d;
systemC_d.tsample= Ts;

figure();
plot(tvec2,systemC_d.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_d mit System_C Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemC_d, 'Faltung Signal_d mit System_C Frequenzbereich');

%% System D Faltungen:
%%signala
system_D_a = conv(sig_a.samples, systemd.samples);
systemD_a.samples = system_D_a;
systemD_a.tsample= Ts;

figure();
plot(tvec2,systemD_a.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_a mit System_D Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemD_a, 'Faltung Signal_a mit System_D Frequenzbereich');

%%signalb
system_D_b = conv(sigb.samples, systemd.samples);
systemD_b.samples = system_D_b;
systemD_b.tsample= Ts;

figure();
plot(tvec2,systemD_b.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_b mit System_D Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemD_b, 'Faltung Signal_b mit System_D Frequenzbereich');

%%signalc
system_D_c = conv(sigc.samples, systemd.samples);
systemD_c.samples = system_D_c;
systemD_c.tsample= Ts;

figure();
plot(tvec2,systemD_c.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_c mit System_D Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemD_c, 'Faltung Signal_c mit System_D Frequenzbereich');

%%signald
system_D_d = conv(sigd.samples, systemd.samples);
systemD_d.samples = system_D_d;
systemD_d.tsample= Ts;

figure();
plot(tvec2,systemD_d.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_d mit System_D Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemD_d, 'Faltung Signal_d mit System_D Frequenzbereich');

%% Funktionen 


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
function [fvec,spec]=plot_signal_spectrum(insig, caption)
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
title(caption);
grid 'on';
grid 'minor';
return;
end