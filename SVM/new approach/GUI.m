function varargout = GUI(varargin)
% GUI MATLAB code for GUI.fig
%      GUI, by itself, creates a new GUI or raises the existing
%      singleton*.
%
%      H = GUI returns the handle to a new GUI or the handle to
%      the existing singleton*.
%
%      GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUI.M with the given input arguments.
%
%      GUI('Property','Value',...) creates a new GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GUI

% Last Modified by GUIDE v2.5 02-Dec-2016 17:40:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before GUI is made visible.
function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to GUI (see VARARGIN)

% Choose default command line output for GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- wdbc.
function checkbox1_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	setop= 1;
    assignin('base','setop',setop)
end


% --- wpbc.
function checkbox5_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	setop =2;
    assignin('base','setop',setop)
end


% --- core yes.
function checkbox6_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	corefeat=1;
    assignin('base','corefeat',corefeat)
end


% --- core no.
function checkbox7_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	corefeat=0;
    assignin('base','corefeat',corefeat)
end


% --- norm yes.
function checkbox8_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	noli =1;
    assignin('base','noli',noli)
end


% --- norm no.
function checkbox9_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	noli =0;
    assignin('base','noli',noli)
end


% --- pca.
function checkbox10_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	featrd=1;
    assignin('base','featrd',featrd)
end


% --- no,thanks.
function checkbox13_Callback(hObject, eventdata, handles)
if (get(hObject,'Value') == 1)
	featrd=0;
    assignin('base','featrd',featrd)
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
noli = evalin('base','noli');
setop = evalin('base','setop');
corefeat = evalin('base','corefeat');
featrd = evalin('base','featrd');
svm_main(noli,setop,corefeat,featrd)
