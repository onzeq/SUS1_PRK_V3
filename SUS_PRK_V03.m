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
%% System A Faltungen:
tvec2 = 0: Ts : 2;
tvec2 = tvec2(1:99999);
system_A_a = conv(sig_a.samples, systema);
systemA_a.samples = system_A_a;
systemA_a.tsample= Ts;


figure();
plot(systemA_a.samples);
dodo=axis;
dodo(1)=-0.015; dodo(2)=0.015;
dodo(3)=0; dodo(4)=1.1;
axis(dodo);
title('Faltung Signal_a mit System_a Zeitbereich');
xlabel('Zeit t in s');
ylabel('Amplitude (ohne Einheit)');

%Erzeugung SpektrumPlot
plot_signal_spectrum(systemA_a, 'Faltung Signal_a mit System_A Frequenzbereich');





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