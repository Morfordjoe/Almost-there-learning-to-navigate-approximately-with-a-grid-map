%Makes predictions for models of grid map navigation
%Can output training, test and model predictions datasets

%Requires function scripts: 
    %function_standardise_range.m
    %function_allcomb.m
    %function_mod_fig.m


File_out = true;
Big_fig_out = false;
model_figs_out = false;


%Home coords - probably always keep at origin here
x_home = 0;
y_home = 0;

limx = 5;
limy = 5;

%x and y values of test dataset
x_vals = linspace(-limx, limx, 101);
y_vals = linspace(-limy, limy, 101);
all_locations = function_allcomb(x_vals, y_vals);
[test_angs, test_dists] = cart2pol(all_locations(:,1), all_locations(:,2));
x = all_locations(:,1);
y = all_locations(:,2);


%figure;
%Index of origin in test dataset
origin_ind = find(ismember(all_locations, [0, 0],'rows'));
%all_locations(origin_ind,:) = [];


%Generate training dataset
train_ang_vals = linspace(0, 2*pi - pi/1000, 180);
train_dist_vals = linspace(0.05, 4, 80);
all_training = function_allcomb(train_ang_vals, train_dist_vals);
[train_x, train_y] = pol2cart(all_training(:,1), all_training(:,2));

Training_TRUE = NaN(length(train_x), 1);
for training_i = 1:length(train_x)
    Training_TRUE(training_i) = atan2((y_home - train_y(training_i)), (x_home - train_x(training_i)));
end

figure;
t = tiledlayout(1,2);
nexttile
scatter(train_x, train_y, 1.5, 'filled');
title('A')
axis equal
xlim([-limx-0.5, limx+0.5])
ylim([-limx-0.5, limx+0.5])
ax = gca;
ax.TitleHorizontalAlignment = 'left';
nexttile
scatter(x, y, 1.5, 'filled');
title('B', 'HorizontalAlignment', 'left')
axis equal
xlim([-limx-0.5, limx+0.5])
ylim([-limx-0.5, limx+0.5])
ax1 = gca;
ax1.TitleHorizontalAlignment = 'left';
%exportgraphics(t,'train_test.png','Resolution',300)


%Make all gradients, and make them vary in same range of numbers - see standardise_range function
denominator_k = 0.1;


%First gradient - varies linearly with y axis
A_base = y;
A = function_standardise_range(A_base, A_base, denominator_k);
A_train_base = train_y;
A_train = function_standardise_range(A_train_base, A_base, denominator_k);

%Second gradient - 4 possible versions

B1_base = x; %Orthogonal grid
B1 = function_standardise_range(B1_base, B1_base, denominator_k);
B1_train_base = train_x;
B1_train = function_standardise_range(B1_train_base, B1_base, denominator_k);

B2i_base = (x - y); %Non-orthogonal grid
B2i = function_standardise_range(B2i_base, B2i_base, denominator_k);
B2i_train_base = (train_x - train_y);
B2i_train = function_standardise_range(B2i_train_base, B2i_base, denominator_k);

B2ii_base = (x - 2*y); %Non-orthogonal grid
B2ii = function_standardise_range(B2ii_base, B2ii_base, denominator_k);
B2ii_train_base = (train_x - 2*train_y);
B2ii_train = function_standardise_range(B2ii_train_base, B2ii_base, denominator_k);


B3i_j1 = 1; 
B3i_j2 = 1;
B3i_base = x - B3i_j1*(y-B3i_j2).*(y-B3i_j2); %Curvilinear
B3i = function_standardise_range(B3i_base, B3i_base, denominator_k);
B3i_train_base = train_x - B3i_j1*(train_y-B3i_j2).*(train_y-B3i_j2); 
B3i_train = function_standardise_range(B3i_train_base, B3i_base, denominator_k);

B3ii_j1 = 0.2; 
B3ii_j2 = -2;
B3ii_base = x - B3ii_j1*(y-B3ii_j2).*(y-B3ii_j2); %Curvilinear
B3ii = function_standardise_range(B3ii_base, B3ii_base, denominator_k);
B3ii_train_base = train_x - B3ii_j1*(train_y-B3ii_j2).*(train_y-B3ii_j2); 
B3ii_train = function_standardise_range(B3ii_train_base, B3ii_base, denominator_k);

B4i_j = 1;
B4i_base = exp(x/B4i_j); %Exponential axis
B4i = function_standardise_range(B4i_base, B4i_base, denominator_k);
B4i_train_base = exp(train_x/B4i_j); %Exponential axis
B4i_train = function_standardise_range(B4i_train_base, B4i_base, denominator_k);

B4ii_j = 6;
B4ii_base = (-x+B4ii_j).*(-x+B4ii_j).*(-x+B4ii_j); 
B4ii = function_standardise_range(B4ii_base, B4ii_base, denominator_k);
B4ii_train_base = (-train_x+B4ii_j).*(-train_x+B4ii_j).*(-train_x+B4ii_j);
B4ii_train = function_standardise_range(B4ii_train_base, B4ii_base, denominator_k);



%The equations for contour plots (with standardisation)
A_e = @(x, y) (y - min(A_base) - range(A_base)*0.5)/(denominator_k*range(A_base));
B1_e = @(x, y) (x - min(B1_base) - range(B1_base)*0.5)/(denominator_k*range(B1_base));
B2i_e = @(x, y) (x - y  - min(B2i_base) - range(B2i_base)*0.5)/(denominator_k*range(B2i_base));
B2ii_e = @(x, y) (x - 2*y  - min(B2ii_base) - range(B2ii_base)*0.5)/(denominator_k*range(B2ii_base));
B3i_e = @(x, y) (x - min(B3i_base) - range(B3i_base)*0.5 - B3i_j1*(y-B3i_j2).*(y-B3i_j2))/(denominator_k*range(B3i_base));
B3ii_e = @(x, y) (x - min(B3ii_base) - range(B3ii_base)*0.5 - B3ii_j1*(y-B3ii_j2).*(y-B3ii_j2))/(denominator_k*range(B3ii_base));
B4i_e = @(x, y) (exp(x/B4i_j) - min(B4i_base) - range(B4i_base)*0.5)/(denominator_k*range(B4i_base));
B4ii_e = @(x, y) ((-x+B4ii_j).*(-x+B4ii_j).*(-x+B4ii_j) - min(B4ii_base) - range(B4ii_base)*0.5)/(denominator_k*range(B4ii_base));


%Correct bicoordinate models

dA_dx = 0;
dA_dy = 1/(denominator_k*range(A_base));

dB1_dx = 1/(denominator_k*range(B1_base));
dB1_dy = 0;

dB2i_dx = 1/(denominator_k*range(B2i_base));
dB2i_dy = -1/(denominator_k*range(B2i_base));

dB2ii_dx = 1/(denominator_k*range(B2ii_base));
dB2ii_dy = -2/(denominator_k*range(B2ii_base));

dB3i_dx = 1/(denominator_k*range(B3i_base));
%dB3i_dy = 2*B3i_j1*(-Y +B3i_j2)/(denominator_k*range(B3i_base));
dB3i_dy_T = (2*B3i_j1*B3i_j2)/(denominator_k*range(B3i_base)); %Target-based

dB3ii_dx = 1/(denominator_k*range(B3ii_base));
%dB3ii_dy = 2*B3ii_j1*(-Y +B3ii_j2)/(denominator_k*range(B3ii_base));
dB3ii_dy_T = (2*B3ii_j1*B3ii_j2)/(denominator_k*range(B3ii_base)); %Target-based

%dB4i_dx = exp(X/B4i_j)/(denominator_k*B4i_j*range(B4i_base));
dB4i_dx_T = 1/(denominator_k*B4i_j*range(B4i_base)); %Target-based
dB4i_dy = 0;

%dB4ii_dx = (-3*B4ii_j.^2 + 6*B4ii_j*X + - 3*X.^2)/(denominator_k*range(B4ii_base));
dB4ii_dx_T = -3*B4ii_j.^2/(denominator_k*range(B4ii_base)); %Target-based
dB4ii_dy = 0;

%Matrices
M_A_B1 = [dA_dx, dA_dy; dB1_dx, dB1_dy];
M_A_B2i = [dA_dx, dA_dy; dB2i_dx, dB2i_dy];
M_A_B2ii = [dA_dx, dA_dy; dB2ii_dx, dB2ii_dy];
M_T_A_B3i = [dA_dx, dA_dy; dB3i_dx, dB3i_dy_T];
M_T_A_B3ii = [dA_dx, dA_dy; dB3ii_dx, dB3ii_dy_T];
M_T_A_B4i = [dA_dx, dA_dy; dB4i_dx_T, dB4i_dy];
M_T_A_B4ii = [dA_dx, dA_dy; dB4ii_dx_T, dB4ii_dy];


%Alphas and betas - angles at which fields vary
%A' and B' - magnitude of how fields vary
%_T to indicate that these are extrapolations at the target (home/origin)
%site. However, A, B1 and B2 are the same wherever the point of
%extrapolation
alpha = atan2(dA_dy, dA_dx);
A_dash = sqrt(dA_dx^2 + dA_dy^2);

beta1 = atan2(dB1_dy, dB1_dx);
B1_dash = sqrt(dB1_dx^2 + dB1_dy^2);

beta2i = atan2(dB2i_dy, dB2i_dx);
B2i_dash = sqrt(dB2i_dx^2 + dB2i_dy^2);

beta2ii = atan2(dB2ii_dy, dB2ii_dx);
B2ii_dash = sqrt(dB2ii_dx^2 + dB2ii_dy^2);

beta3i_T = atan2(dB3i_dy_T, dB3i_dx);
B3i_dash_T = sqrt(dB3i_dx^2 + dB3i_dy_T^2);

beta3ii_T = atan2(dB3ii_dy_T, dB3ii_dx);
B3ii_dash_T = sqrt(dB3ii_dx^2 + dB3ii_dy_T^2);

beta4i_T = atan2(dB4i_dy, dB4i_dx_T);
B4i_dash_T = sqrt(dB4i_dx_T^2 + dB4i_dy^2);

beta4ii_T = atan2(dB4ii_dy, dB4ii_dx_T);
B4ii_dash_T = sqrt(dB4ii_dx_T^2 + dB4ii_dy^2);

%%%%%%%

TRUE = NaN(length(x), 1);

A_B1_CORRECT = NaN(length(x), 1);
A_B2i_CORRECT = NaN(length(x), 1);
A_B2ii_CORRECT = NaN(length(x), 1);
A_B3i_CORRECT_T = NaN(length(x), 1);
A_B3ii_CORRECT_T = NaN(length(x), 1);
A_B4i_CORRECT_T = NaN(length(x), 1);
A_B4ii_CORRECT_T = NaN(length(x), 1);

A_B3i_CORRECT_R = NaN(length(x), 1);
A_B3ii_CORRECT_R = NaN(length(x), 1);
A_B4i_CORRECT_R = NaN(length(x), 1);
A_B4ii_CORRECT_R = NaN(length(x), 1);

A_B3i_CORRECT_Train = NaN(length(x), 1);
A_B3ii_CORRECT_Train = NaN(length(x), 1);
A_B4i_CORRECT_Train = NaN(length(x), 1);
A_B4ii_CORRECT_Train = NaN(length(x), 1);

A_B1_APPROX = NaN(length(x), 1);
A_B2i_APPROX = NaN(length(x), 1);
A_B2ii_APPROX = NaN(length(x), 1);
A_B3i_APPROX_T = NaN(length(x), 1);
A_B3ii_APPROX_T = NaN(length(x), 1);
A_B4i_APPROX_T = NaN(length(x), 1);
A_B4ii_APPROX_T = NaN(length(x), 1);

A_B3i_APPROX_R = NaN(length(x), 1);
A_B3ii_APPROX_R = NaN(length(x), 1);
A_B4i_APPROX_R = NaN(length(x), 1);
A_B4ii_APPROX_R = NaN(length(x), 1);

A_B3i_APPROX_Train = NaN(length(x), 1);
A_B3ii_APPROX_Train = NaN(length(x), 1);
A_B4i_APPROX_Train = NaN(length(x), 1);
A_B4ii_APPROX_Train = NaN(length(x), 1);


A_B1_DIREC = NaN(length(x), 1);
A_B2i_DIREC = NaN(length(x), 1);
A_B2ii_DIREC = NaN(length(x), 1);
A_B3i_DIREC_T = NaN(length(x), 1);
A_B3ii_DIREC_T = NaN(length(x), 1);
A_B4i_DIREC_T = NaN(length(x), 1);
A_B4ii_DIREC_T = NaN(length(x), 1);

A_B3i_DIREC_R = NaN(length(x), 1);
A_B3ii_DIREC_R = NaN(length(x), 1);
A_B4i_DIREC_R = NaN(length(x), 1);
A_B4ii_DIREC_R = NaN(length(x), 1);

A_B3i_DIREC_Train = NaN(length(x), 1);
A_B3ii_DIREC_Train = NaN(length(x), 1);
A_B4i_DIREC_Train = NaN(length(x), 1);
A_B4ii_DIREC_Train = NaN(length(x), 1);

%%%%%%%


beta3i_Train = NaN(length(train_x), 1);
B3i_dash_Train = NaN(length(train_x), 1);
beta3ii_Train = NaN(length(train_x), 1);
B3ii_dash_Train = NaN(length(train_x), 1);
beta4i_Train = NaN(length(train_x), 1);
B4i_dash_Train = NaN(length(train_x), 1);
beta4ii_Train = NaN(length(train_x), 1);
B4ii_dash_Train = NaN(length(train_x), 1);



for train_i = 1:length(train_x)
    dB3i_dy_Train = 2*B3i_j1*(-train_y(train_i) +B3i_j2)/(denominator_k*range(B3i_base));
    dB3ii_dy_Train = 2*B3ii_j1*(-train_y(train_i) +B3ii_j2)/(denominator_k*range(B3ii_base));
    dB4i_dx_Train = exp(train_x(train_i)/B4i_j)/(B4i_j*denominator_k*range(B4i_base));
    dB4ii_dx_Train = (-3*B4ii_j.^2 + 6*B4ii_j*train_x(train_i) + - 3*train_x(train_i).^2)/(denominator_k*range(B4ii_base));

    beta3i_Train(train_i) = atan2(dB3i_dy_Train, dB3i_dx);
    beta3ii_Train(train_i) = atan2(dB3ii_dy_Train, dB3ii_dx);
    B3i_dash_Train(train_i) = sqrt(dB3i_dx^2 + dB3i_dy_Train^2);
    B3ii_dash_Train(train_i) = sqrt(dB3ii_dx^2 + dB3ii_dy_Train^2);

    beta4i_Train(train_i) = atan2(dB4i_dy, dB4i_dx_Train);
    B4i_dash_Train(train_i) = sqrt(dB4i_dx_Train^2 + dB4i_dy^2);

    beta4ii_Train(train_i) = atan2(dB4ii_dy, dB4ii_dx_Train);
    B4ii_dash_Train(train_i) = sqrt(dB4ii_dx_Train^2 + dB4ii_dy^2);

end


cmean_beta3i_Train = atan2(mean(sin(beta3i_Train),1),mean(cos(beta3i_Train),1));
mean_B3i_dash_Train = mean(B3i_dash_Train);
cmean_beta3ii_Train = atan2(mean(sin(beta3ii_Train),1),mean(cos(beta3ii_Train),1));
mean_B3ii_dash_Train = mean(B3ii_dash_Train);
cmean_beta4i_Train = atan2(mean(sin(beta4i_Train),1),mean(cos(beta4i_Train),1));
mean_B4i_dash_Train = mean(B4i_dash_Train);
cmean_beta4ii_Train = atan2(mean(sin(beta4ii_Train),1),mean(cos(beta4ii_Train),1));
mean_B4ii_dash_Train = mean(B4ii_dash_Train);


for release_i = 1:length(x)

    if release_i == origin_ind
        continue
    end

    %CORRECT BICOORDINATE TARGET MODEL  
    xP_yP_A_B1 = (M_A_B1^(-1))*[A(release_i) - A(origin_ind); B1(release_i) - B1(origin_ind)] + [x_home; y_home];
    xP_yP_A_B2i = (M_A_B2i^(-1))*[A(release_i) - A(origin_ind); B2i(release_i) - B2i(origin_ind)] + [x_home; y_home];
    xP_yP_A_B2ii = (M_A_B2ii^(-1))*[A(release_i) - A(origin_ind); B2ii(release_i) - B2ii(origin_ind)] + [x_home; y_home];
    xP_yP_A_B3i_T = (M_T_A_B3i^(-1))*[A(release_i) - A(origin_ind); B3i(release_i) - B3i(origin_ind)] + [x_home; y_home];
    xP_yP_A_B3ii_T = (M_T_A_B3ii^(-1))*[A(release_i) - A(origin_ind); B3ii(release_i) - B3ii(origin_ind)] + [x_home; y_home];
    xP_yP_A_B4i_T = (M_T_A_B4i^(-1))*[A(release_i) - A(origin_ind); B4i(release_i) - B4i(origin_ind)] + [x_home; y_home];
    xP_yP_A_B4ii_T = (M_T_A_B4ii^(-1))*[A(release_i) - A(origin_ind); B4ii(release_i) - B4ii(origin_ind)] + [x_home; y_home];


    TRUE(release_i) = atan2((y_home - y(release_i)), (x_home - x(release_i)));

    A_B1_CORRECT(release_i) = atan2((y_home - xP_yP_A_B1(2)),(x_home - xP_yP_A_B1(1)));
    A_B2i_CORRECT(release_i) = atan2((y_home - xP_yP_A_B2i(2)),(x_home - xP_yP_A_B2i(1)));
    A_B2ii_CORRECT(release_i) = atan2((y_home - xP_yP_A_B2ii(2)),(x_home - xP_yP_A_B2ii(1)));
    A_B3i_CORRECT_T(release_i) = atan2((y_home - xP_yP_A_B3i_T(2)),(x_home - xP_yP_A_B3i_T(1)));
    A_B3ii_CORRECT_T(release_i) = atan2((y_home - xP_yP_A_B3ii_T(2)),(x_home - xP_yP_A_B3ii_T(1)));
    A_B4i_CORRECT_T(release_i) = atan2((y_home - xP_yP_A_B4i_T(2)),(x_home - xP_yP_A_B4i_T(1)));
    A_B4ii_CORRECT_T(release_i) = atan2((y_home - xP_yP_A_B4ii_T(2)),(x_home - xP_yP_A_B4ii_T(1)));
    

    %Check - different method of CORRECT COORDINATE TARGET
    %Currently just checking B3i version
    %dx_dy_A_B3i_Correct_T = [(((A(origin_ind) - A(release_i))*cos(beta3i_T - pi/2))/(A_dash*sin(beta3i_T - alpha))) +  ...
        %(((B3i(origin_ind) - B3i(release_i))*cos(alpha_T - pi/2))/(B3i_dash_T*sin(alpha_T - beta3i_T)));
        %(((A(origin_ind) - A(release_i))*sin(beta3i_T - pi/2))/(A_dash*sin(beta3i_T - alpha))) +  ...
        %(((B3i(origin_ind) - B3i(release_i))*sin(alpha - pi/2))/(B3i_dash*sin(alpha - beta3i)))];
    %atan2(dx_dy_A_B3i_Correct_T(2), dx_dy_A_B3i_Correct_T(1));


    %CORRECT BICOORDINATE RELEASE MODEL  

    dB3i_dy_R = 2*B3i_j1*(-y(release_i) +B3i_j2)/(denominator_k*range(B3i_base));
    dB3ii_dy_R = 2*B3ii_j1*(-y(release_i) +B3ii_j2)/(denominator_k*range(B3ii_base));
    dB4i_dx_R = exp(x(release_i)/B4i_j)/(B4i_j*denominator_k*range(B4i_base));
    dB4ii_dx_R = (-3*B4ii_j.^2 + 6*B4ii_j*x(release_i) + - 3*x(release_i).^2)/(denominator_k*range(B4ii_base));

    M_R_A_B3i = [dA_dx, dA_dy; dB3i_dx, dB3i_dy_R];
    M_R_A_B3ii = [dA_dx, dA_dy; dB3ii_dx, dB3ii_dy_R];
    M_R_A_B4i = [dA_dx, dA_dy; dB4i_dx_R, dB4i_dy];
    M_R_A_B4ii = [dA_dx, dA_dy; dB4ii_dx_R, dB4ii_dy];    
    
    xP_yP_A_B3i_R = (M_R_A_B3i^(-1))*[A(release_i) - A(origin_ind); B3i(release_i) - B3i(origin_ind)] + [x_home; y_home];
    xP_yP_A_B3ii_R = (M_R_A_B3ii^(-1))*[A(release_i) - A(origin_ind); B3ii(release_i) - B3ii(origin_ind)] + [x_home; y_home];
    xP_yP_A_B4i_R = (M_R_A_B4i^(-1))*[A(release_i) - A(origin_ind); B4i(release_i) - B4i(origin_ind)] + [x_home; y_home];
    xP_yP_A_B4ii_R = (M_R_A_B4ii^(-1))*[A(release_i) - A(origin_ind); B4ii(release_i) - B4ii(origin_ind)] + [x_home; y_home];
    
    A_B3i_CORRECT_R(release_i) = atan2((y_home - xP_yP_A_B3i_R(2)),(x_home - xP_yP_A_B3i_R(1)));
    A_B3ii_CORRECT_R(release_i) = atan2((y_home - xP_yP_A_B3ii_R(2)),(x_home - xP_yP_A_B3ii_R(1)));
    A_B4i_CORRECT_R(release_i) = atan2((y_home - xP_yP_A_B4i_R(2)),(x_home - xP_yP_A_B4i_R(1)));
    A_B4ii_CORRECT_R(release_i) = atan2((y_home - xP_yP_A_B4ii_R(2)),(x_home - xP_yP_A_B4ii_R(1)));

    %CORRECT BICOORDINATE TRAIN MODEL

    dx_dy_A_B3i_Correct_Train = [(((A(origin_ind) - A(release_i))*cos(cmean_beta3i_Train - pi/2))/(A_dash*sin(cmean_beta3i_Train - alpha))) +  ...
        (((B3i(origin_ind) - B3i(release_i))*cos(alpha - pi/2))/(mean_B3i_dash_Train*sin(alpha - cmean_beta3i_Train)));
        (((A(origin_ind) - A(release_i))*sin(cmean_beta3i_Train - pi/2))/(A_dash*sin(cmean_beta3i_Train - alpha))) +  ...
        (((B3i(origin_ind) - B3i(release_i))*sin(alpha - pi/2))/(mean_B3i_dash_Train*sin(alpha - cmean_beta3i_Train)))];
    A_B3i_CORRECT_Train(release_i) = atan2(dx_dy_A_B3i_Correct_Train(2), dx_dy_A_B3i_Correct_Train(1));
        
    dx_dy_A_B3ii_Correct_Train = [(((A(origin_ind) - A(release_i))*cos(cmean_beta3ii_Train - pi/2))/(A_dash*sin(cmean_beta3ii_Train - alpha))) +  ...
        (((B3ii(origin_ind) - B3ii(release_i))*cos(alpha - pi/2))/(mean_B3ii_dash_Train*sin(alpha - cmean_beta3ii_Train)));
        (((A(origin_ind) - A(release_i))*sin(cmean_beta3ii_Train - pi/2))/(A_dash*sin(cmean_beta3ii_Train - alpha))) +  ...
        (((B3ii(origin_ind) - B3ii(release_i))*sin(alpha - pi/2))/(mean_B3ii_dash_Train*sin(alpha - cmean_beta3ii_Train)))];
    A_B3ii_CORRECT_Train(release_i) = atan2(dx_dy_A_B3ii_Correct_Train(2), dx_dy_A_B3ii_Correct_Train(1));


    dx_dy_A_B4i_Correct_Train = [(((A(origin_ind) - A(release_i))*cos(cmean_beta4i_Train - pi/2))/(A_dash*sin(cmean_beta4i_Train - alpha))) +  ...
        (((B4i(origin_ind) - B4i(release_i))*cos(alpha - pi/2))/(mean_B4i_dash_Train*sin(alpha - cmean_beta4i_Train)));
        (((A(origin_ind) - A(release_i))*sin(cmean_beta4i_Train - pi/2))/(A_dash*sin(cmean_beta4i_Train - alpha))) +  ...
        (((B4i(origin_ind) - B4i(release_i))*sin(alpha - pi/2))/(mean_B4i_dash_Train*sin(alpha - cmean_beta4i_Train)))];
    A_B4i_CORRECT_Train(release_i) = atan2(dx_dy_A_B4i_Correct_Train(2), dx_dy_A_B4i_Correct_Train(1));

    dx_dy_A_B4ii_Correct_Train = [(((A(origin_ind) - A(release_i))*cos(cmean_beta4ii_Train - pi/2))/(A_dash*sin(cmean_beta4ii_Train - alpha))) +  ...
        (((B4ii(origin_ind) - B4ii(release_i))*cos(alpha - pi/2))/(mean_B4ii_dash_Train*sin(alpha - cmean_beta4ii_Train)));
        (((A(origin_ind) - A(release_i))*sin(cmean_beta4ii_Train - pi/2))/(A_dash*sin(cmean_beta4ii_Train - alpha))) +  ...
        (((B4ii(origin_ind) - B4ii(release_i))*sin(alpha - pi/2))/(mean_B4ii_dash_Train*sin(alpha - cmean_beta4ii_Train)))];
    A_B4ii_CORRECT_Train(release_i) = atan2(dx_dy_A_B4ii_Correct_Train(2), dx_dy_A_B4ii_Correct_Train(1));


    %APPROXIMATE BICOORDINATE (TARGET-BASED)
    
    dx_dy_A_B1_Approx = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B1(origin_ind) - B1(release_i))*cos(beta1))/B1_dash;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B1(origin_ind) - B1(release_i))*sin(beta1))/B1_dash];
    A_B1_APPROX(release_i) = atan2(dx_dy_A_B1_Approx(2), dx_dy_A_B1_Approx(1));
    %Should give same answer as correct bicoordinate target for B1
    
    dx_dy_A_B2i_Approx = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B2i(origin_ind) - B2i(release_i))*cos(beta2i))/B2i_dash;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B2i(origin_ind) - B2i(release_i))*sin(beta2i))/B2i_dash];
    A_B2i_APPROX(release_i) = atan2(dx_dy_A_B2i_Approx(2), dx_dy_A_B2i_Approx(1));
    
    dx_dy_A_B2ii_Approx = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B2ii(origin_ind) - B2ii(release_i))*cos(beta2ii))/B2ii_dash;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B2ii(origin_ind) - B2ii(release_i))*sin(beta2ii))/B2ii_dash];
    A_B2ii_APPROX(release_i) = atan2(dx_dy_A_B2ii_Approx(2), dx_dy_A_B2ii_Approx(1));
    
    dx_dy_A_B3i_Approx_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B3i(origin_ind) - B3i(release_i))*cos(beta3i_T))/B3i_dash_T;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B3i(origin_ind) - B3i(release_i))*sin(beta3i_T))/B3i_dash_T];
    A_B3i_APPROX_T(release_i) = atan2(dx_dy_A_B3i_Approx_T(2), dx_dy_A_B3i_Approx_T(1));

    dx_dy_A_B3ii_Approx_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B3ii(origin_ind) - B3ii(release_i))*cos(beta3ii_T))/B3ii_dash_T;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B3ii(origin_ind) - B3ii(release_i))*sin(beta3ii_T))/B3ii_dash_T];
    A_B3ii_APPROX_T(release_i) = atan2(dx_dy_A_B3ii_Approx_T(2), dx_dy_A_B3ii_Approx_T(1));
    
    dx_dy_A_B4i_Approx_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B4i(origin_ind) - B4i(release_i))*cos(beta4i_T))/B4i_dash_T;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B4i(origin_ind) - B4i(release_i))*sin(beta4i_T))/B4i_dash_T];
    A_B4i_APPROX_T(release_i) = atan2(dx_dy_A_B4i_Approx_T(2), dx_dy_A_B4i_Approx_T(1));
    %Should give same answer as correct bicoordinate target for B4

    dx_dy_A_B4ii_Approx_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B4ii(origin_ind) - B4ii(release_i))*cos(beta4ii_T))/B4ii_dash_T;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B4ii(origin_ind) - B4ii(release_i))*sin(beta4ii_T))/B4ii_dash_T];
    A_B4ii_APPROX_T(release_i) = atan2(dx_dy_A_B4ii_Approx_T(2), dx_dy_A_B4ii_Approx_T(1));

    %APPROXIMATE BICOORDINATE (RELEASE-BASED)

    beta3i_R = atan2(dB3i_dy_R, dB3i_dx);
    B3i_dash_R = sqrt(dB3i_dx^2 + dB3i_dy_R^2);

    beta3ii_R = atan2(dB3ii_dy_R, dB3ii_dx);
    B3ii_dash_R = sqrt(dB3ii_dx^2 + dB3ii_dy_R^2);

    beta4i_R = atan2(dB4i_dy, dB4i_dx_R);
    B4i_dash_R = sqrt(dB4i_dx_R^2 + dB4i_dy^2);

    beta4ii_R = atan2(dB4ii_dy, dB4ii_dx_R);
    B4ii_dash_R = sqrt(dB4ii_dx_R^2 + dB4ii_dy^2);


    dx_dy_A_B3i_Approx_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B3i(origin_ind) - B3i(release_i))*cos(beta3i_R))/B3i_dash_R;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B3i(origin_ind) - B3i(release_i))*sin(beta3i_R))/B3i_dash_R];
    A_B3i_APPROX_R(release_i) = atan2(dx_dy_A_B3i_Approx_R(2), dx_dy_A_B3i_Approx_R(1));

    dx_dy_A_B3ii_Approx_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B3ii(origin_ind) - B3ii(release_i))*cos(beta3ii_R))/B3ii_dash_R;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B3ii(origin_ind) - B3ii(release_i))*sin(beta3ii_R))/B3ii_dash_R];
    A_B3ii_APPROX_R(release_i) = atan2(dx_dy_A_B3ii_Approx_R(2), dx_dy_A_B3ii_Approx_R(1));

    dx_dy_A_B4i_Approx_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B4i(origin_ind) - B4i(release_i))*cos(beta4i_R))/B4i_dash_R;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B4i(origin_ind) - B4i(release_i))*sin(beta4i_R))/B4i_dash_R];
    A_B4i_APPROX_R(release_i) = atan2(dx_dy_A_B4i_Approx_R(2), dx_dy_A_B4i_Approx_R(1));
    
    dx_dy_A_B4ii_Approx_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B4ii(origin_ind) - B4ii(release_i))*cos(beta4ii_R))/B4ii_dash_R;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B4ii(origin_ind) - B4ii(release_i))*sin(beta4ii_R))/B4ii_dash_R];
    A_B4ii_APPROX_R(release_i) = atan2(dx_dy_A_B4ii_Approx_R(2), dx_dy_A_B4ii_Approx_R(1));


    %APPROXIMATE BICOORDINATE (TRAIN MODEL)
    
    dx_dy_A_B3i_Approx_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B3i(origin_ind) - B3i(release_i))*cos(cmean_beta3i_Train))/mean_B3i_dash_Train;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B3i(origin_ind) - B3i(release_i))*sin(cmean_beta3i_Train))/mean_B3i_dash_Train];
    A_B3i_APPROX_Train(release_i) = atan2(dx_dy_A_B3i_Approx_Train(2), dx_dy_A_B3i_Approx_Train(1));

    dx_dy_A_B3ii_Approx_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B3ii(origin_ind) - B3ii(release_i))*cos(cmean_beta3ii_Train))/mean_B3ii_dash_Train;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B3ii(origin_ind) - B3ii(release_i))*sin(cmean_beta3ii_Train))/mean_B3ii_dash_Train];
    A_B3ii_APPROX_Train(release_i) = atan2(dx_dy_A_B3ii_Approx_Train(2), dx_dy_A_B3ii_Approx_Train(1));

    dx_dy_A_B4i_Approx_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B4i(origin_ind) - B4i(release_i))*cos(cmean_beta4i_Train))/mean_B4i_dash_Train;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B4i(origin_ind) - B4i(release_i))*sin(cmean_beta4i_Train))/mean_B4i_dash_Train];
    A_B4i_APPROX_Train(release_i) = atan2(dx_dy_A_B4i_Approx_Train(2), dx_dy_A_B4i_Approx_Train(1));
    
    dx_dy_A_B4ii_Approx_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_dash + ...
        ((B4ii(origin_ind) - B4ii(release_i))*cos(cmean_beta4ii_Train))/mean_B4ii_dash_Train;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_dash + ...
        ((B4ii(origin_ind) - B4ii(release_i))*sin(cmean_beta4ii_Train))/mean_B4ii_dash_Train];
    A_B4ii_APPROX_Train(release_i) = atan2(dx_dy_A_B4ii_Approx_Train(2), dx_dy_A_B4ii_Approx_Train(1));


    %DIRECTIONAL (TARGET-BASED)

    if abs(A(origin_ind) - A(release_i))==0
        A_denom = 1;
    else
        A_denom = abs(A(origin_ind) - A(release_i));
    end
    if abs(B1(origin_ind) - B1(release_i))==0
        B1_denom = 1;
    else
        B1_denom = abs(B1(origin_ind) - B1(release_i));
    end
    if abs(B2i(origin_ind) - B2i(release_i))==0
        B2i_denom = 1;
    else
        B2i_denom = abs(B2i(origin_ind) - B2i(release_i));
    end
    if abs(B2ii(origin_ind) - B2ii(release_i))==0
        B2ii_denom = 1;
    else
        B2ii_denom = abs(B2ii(origin_ind) - B2ii(release_i));
    end
    if abs(B3i(origin_ind) - B3i(release_i))==0
        B3i_denom = 1;
    else
        B3i_denom = abs(B3i(origin_ind) - B3i(release_i));
    end
    if abs(B3ii(origin_ind) - B3ii(release_i))==0
        B3ii_denom = 1;
    else
        B3ii_denom = abs(B3ii(origin_ind) - B3ii(release_i));
    end
    if abs(B4i(origin_ind) - B4i(release_i))==0
        B4i_denom = 1;
    else
        B4i_denom = abs(B4i(origin_ind) - B4i(release_i));
    end
    if abs(B4ii(origin_ind) - B4ii(release_i))==0
        B4ii_denom = 1;
    else
        B4ii_denom = abs(B4ii(origin_ind) - B4ii(release_i));
    end
    
    dx_dy_A_B1_Directional = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B1(origin_ind) - B1(release_i))*cos(beta1))/B1_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B1(origin_ind) - B1(release_i))*sin(beta1))/B1_denom];
    A_B1_DIREC(release_i) = atan2(dx_dy_A_B1_Directional(2), dx_dy_A_B1_Directional(1));
    
    dx_dy_A_B2i_Directional = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B2i(origin_ind) - B2i(release_i))*cos(beta2i))/B2i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B2i(origin_ind) - B2i(release_i))*sin(beta2i))/B2i_denom];
    A_B2i_DIREC(release_i) = atan2(dx_dy_A_B2i_Directional(2), dx_dy_A_B2i_Directional(1));
    
    dx_dy_A_B2ii_Directional = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B2ii(origin_ind) - B2ii(release_i))*cos(beta2ii))/B2ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B2ii(origin_ind) - B2ii(release_i))*sin(beta2ii))/B2ii_denom];
    A_B2ii_DIREC(release_i) = atan2(dx_dy_A_B2ii_Directional(2), dx_dy_A_B2ii_Directional(1));

    dx_dy_A_B3i_Directional_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B3i(origin_ind) - B3i(release_i))*cos(beta3i_T))/B3i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B3i(origin_ind) - B3i(release_i))*sin(beta3i_T))/B3i_denom];
    A_B3i_DIREC_T(release_i) = atan2(dx_dy_A_B3i_Directional_T(2), dx_dy_A_B3i_Directional_T(1));

    dx_dy_A_B3ii_Directional_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B3ii(origin_ind) - B3ii(release_i))*cos(beta3ii_T))/B3ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B3ii(origin_ind) - B3ii(release_i))*sin(beta3ii_T))/B3ii_denom];
    A_B3ii_DIREC_T(release_i) = atan2(dx_dy_A_B3ii_Directional_T(2), dx_dy_A_B3ii_Directional_T(1));

    dx_dy_A_B4i_Directional_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B4i(origin_ind) - B4i(release_i))*cos(beta4i_T))/B4i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B4i(origin_ind) - B4i(release_i))*sin(beta4i_T))/B4i_denom];
    A_B4i_DIREC_T(release_i) = atan2(dx_dy_A_B4i_Directional_T(2), dx_dy_A_B4i_Directional_T(1));
    
    dx_dy_A_B4ii_Directional_T = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B4ii(origin_ind) - B4ii(release_i))*cos(beta4ii_T))/B4ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B4ii(origin_ind) - B4ii(release_i))*sin(beta4ii_T))/B4ii_denom];
    A_B4ii_DIREC_T(release_i) = atan2(dx_dy_A_B4ii_Directional_T(2), dx_dy_A_B4ii_Directional_T(1));


    %DIRECTIONAL (RELEASE-BASED)

    dx_dy_A_B3i_Directional_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B3i(origin_ind) - B3i(release_i))*cos(beta3i_R))/B3i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B3i(origin_ind) - B3i(release_i))*sin(beta3i_R))/B3i_denom];
    A_B3i_DIREC_R(release_i) = atan2(dx_dy_A_B3i_Directional_R(2), dx_dy_A_B3i_Directional_R(1));

    dx_dy_A_B3ii_Directional_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B3ii(origin_ind) - B3ii(release_i))*cos(beta3ii_R))/B3ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B3ii(origin_ind) - B3ii(release_i))*sin(beta3ii_R))/B3ii_denom];
    A_B3ii_DIREC_R(release_i) = atan2(dx_dy_A_B3ii_Directional_R(2), dx_dy_A_B3ii_Directional_R(1));
    
    dx_dy_A_B4i_Directional_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B4i(origin_ind) - B4i(release_i))*cos(beta4i_R))/B4i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B4i(origin_ind) - B4i(release_i))*sin(beta4i_R))/B4i_denom];
    A_B4i_DIREC_R(release_i) = atan2(dx_dy_A_B4i_Directional_R(2), dx_dy_A_B4i_Directional_R(1));

    dx_dy_A_B4ii_Directional_R = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B4ii(origin_ind) - B4ii(release_i))*cos(beta4ii_R))/B4ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B4ii(origin_ind) - B4ii(release_i))*sin(beta4ii_R))/B4ii_denom];
    A_B4ii_DIREC_R(release_i) = atan2(dx_dy_A_B4ii_Directional_R(2), dx_dy_A_B4ii_Directional_R(1));


    %DIRECTIONAL (TRAIN-BASED)

    dx_dy_A_B3i_Directional_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B3i(origin_ind) - B3i(release_i))*cos(cmean_beta3i_Train))/B3i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B3i(origin_ind) - B3i(release_i))*sin(cmean_beta3i_Train))/B3i_denom];
    A_B3i_DIREC_Train(release_i) = atan2(dx_dy_A_B3i_Directional_Train(2), dx_dy_A_B3i_Directional_Train(1));

    dx_dy_A_B3ii_Directional_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B3ii(origin_ind) - B3ii(release_i))*cos(cmean_beta3ii_Train))/B3ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B3ii(origin_ind) - B3ii(release_i))*sin(cmean_beta3ii_Train))/B3ii_denom];
    A_B3ii_DIREC_Train(release_i) = atan2(dx_dy_A_B3ii_Directional_Train(2), dx_dy_A_B3ii_Directional_Train(1));
    
    dx_dy_A_B4i_Directional_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B4i(origin_ind) - B4i(release_i))*cos(cmean_beta4i_Train))/B4i_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B4i(origin_ind) - B4i(release_i))*sin(cmean_beta4i_Train))/B4i_denom];
    A_B4i_DIREC_Train(release_i) = atan2(dx_dy_A_B4i_Directional_Train(2), dx_dy_A_B4i_Directional_Train(1));

    dx_dy_A_B4ii_Directional_Train = [((A(origin_ind) - A(release_i))*cos(alpha))/A_denom + ...
        ((B4ii(origin_ind) - B4ii(release_i))*cos(cmean_beta4ii_Train))/B4ii_denom;
        ((A(origin_ind) - A(release_i))*sin(alpha))/A_denom + ...
        ((B4ii(origin_ind) - B4ii(release_i))*sin(cmean_beta4ii_Train))/B4ii_denom];
    A_B4ii_DIREC_Train(release_i) = atan2(dx_dy_A_B4ii_Directional_Train(2), dx_dy_A_B4ii_Directional_Train(1));

end

%Training data
TRAIN_TABLE = table(all_training(:,1), all_training(:,2), train_x, train_y, A_train, B1_train, B2i_train, B2ii_train, B3i_train, B3ii_train, B4i_train, B4ii_train, Training_TRUE, ...
    'VariableNames', {'Angs', 'Dists', 'X', 'Y', 'A', 'B1', 'B2i', 'B2ii', 'B3i', 'B3ii', 'B4i', 'B4ii', 'Training_TRUE'});

%Testing data
TEST_TABLE = table(test_angs, test_dists, x, y, A, B1, B2i, B2ii, B3i, B3ii, B4i, B4ii, TRUE, ...
    'VariableNames', {'Angs', 'Dists', 'X', 'Y', 'A', 'B1', 'B2i', 'B2ii', 'B3i', 'B3ii', 'B4i', 'B4ii','TRUE'});
TEST_TABLE(origin_ind, :) = [];

%Model predictions
OUT_TABLE = table(TRUE, ...
    A_B3i_CORRECT_T, A_B3ii_CORRECT_T, A_B4i_CORRECT_T, A_B4ii_CORRECT_T, ...
    A_B3i_CORRECT_R, A_B3ii_CORRECT_R, A_B4i_CORRECT_R, A_B4ii_CORRECT_R, ...
    A_B3i_CORRECT_Train, A_B3ii_CORRECT_Train, A_B4i_CORRECT_Train, A_B4ii_CORRECT_Train, ...
    A_B2i_APPROX, A_B2ii_APPROX, A_B3i_APPROX_T, A_B3ii_APPROX_T, ...
    A_B3i_APPROX_R, A_B3ii_APPROX_R, ...
    A_B3i_APPROX_Train, A_B3ii_APPROX_Train, ...
    A_B1_DIREC, A_B2i_DIREC, A_B2ii_DIREC, A_B3i_DIREC_T, A_B3ii_DIREC_T, A_B4i_DIREC_T, A_B4ii_DIREC_T, ...
    A_B3i_DIREC_R, A_B3ii_DIREC_R, ...
    A_B3i_DIREC_Train, A_B3ii_DIREC_Train, ...
    'VariableNames', {'TRUE', ...
    'A_B3i_CORRECT_T', 'A_B3ii_CORRECT_T', 'A_B4i_CORRECT/APPROX_T', 'A_B4ii_CORRECT/APPROX_T', ...
    'A_B3i_CORRECT_R', 'A_B3ii_CORRECT_R', 'A_B4i_CORRECT/APPROX_R', 'A_B4ii_CORRECT/APPROX_R', ...
    'A_B3i_CORRECT_Train', 'A_B3ii_CORRECT_Train', 'A_B4i_CORRECT/APPROX_Train', 'A_B4ii_CORRECT/APPROX_Train', ...
    'A_B2i_APPROX', 'A_B2ii_APPROX', 'A_B3i_APPROX_T', 'A_B3ii_APPROX_T', ...
    'A_B3i_APPROX_R', 'A_B3ii_APPROX_R', ...
    'A_B3i_APPROX_Train', 'A_B3ii_APPROX_Train', ...
    'A_B1_DIREC', 'A_B2i_DIREC', 'A_B2ii_DIREC', 'A_B3i_DIREC_T', 'A_B3ii_DIREC_T', 'A_B4i_DIREC', 'A_B4ii_DIREC', ...
    'A_B3i_DIREC_R', 'A_B3ii_DIREC_R', ...
    'A_B3i_DIREC_Train', 'A_B3ii_DIREC_Train'});
OUT_TABLE(origin_ind, :) = [];


if File_out
    writetable(TRAIN_TABLE,'Training_data.csv');
    writetable(TEST_TABLE,'Testing_data.csv');
    writetable(OUT_TABLE,'All_model_predictions.csv');
end

%Produces figure of all gradient environments
figure('Position', [1, 1, 600, 300])
big_fig = tiledlayout(2, 4, 'TileSpacing','Compact','Padding','Compact');
nexttile
scatter(x, y, 0.5, [0 0.4470 0.7410], 'filled');
hold on
scatter(train_x, train_y, 0.75,'magenta', 'filled');
plot(0, 0, '.', 'Color', 'black', 'MarkerSize', 25)
axis equal
title('\color{magenta}Training\color{black}/\color[rgb]{0 0.4470 0.7410}Test')
xlim([-limx-0.5, limx+0.5])
ylim([-limx-0.5, limx+0.5])
ax = gca;
ax.TitleHorizontalAlignment = 'left';
hold off
list_grads = {B1_e, B2i_e, B2ii_e, B3i_e, B3ii_e, B4i_e, B4ii_e};
title_grads = ["\color{blue}A\color{black}/\color{red}B1", "\color{blue}A\color{black}/\color{red}B2i", ...
    "\color{blue}A\color{black}/\color{red}B2ii", "\color{blue}A\color{black}/\color{red}B3i", ...
    "\color{blue}A\color{black}/\color{red}B3ii", "\color{blue}A\color{black}/\color{red}B4i", ...
    "\color{blue}A\color{black}/\color{red}B4ii"];
for i = [2,4,6,1,3,5,7]
    fcontour_e = list_grads{i};
    nexttile
    hold on
    fcontour(A_e, [-limx-0.51, limx+0.51, -limy-0.51, limy+0.51], 'LineWidth',0.5, 'LineColor', 'blue', 'LevelList', [-10:-1, 1:10])
    fcontour(fcontour_e, [-limx-0.51, limx+0.51, -limy-0.51, limy+0.51], '--', 'LineWidth',0.5, 'LineColor', 'red', 'LevelList', [-10:-1, 1:10])
    axis equal
    title(title_grads(i))
    xlim([-limx-0.5, limx+0.5])
    ylim([-limy-0.5, limy+0.5])
    ax = gca;
    ax.TitleHorizontalAlignment = 'left';
    fcontour(A_e, [-limx-0.51, limx+0.51, -limy-0.51, limy+0.51], 'LineWidth',1.5, 'LineColor', 'blue', 'LevelList', [0])
    fcontour(fcontour_e, [-limx-0.51, limx+0.51, -limy-0.51, limy+0.51], '--', 'LineWidth',1.5, 'LineColor', 'red', 'LevelList', [0])
    plot(0, 0, '.', 'Color', 'black', 'MarkerSize', 25)
    hold off
end

if Big_fig_out
    exportgraphics(big_fig,'Write up/big_fig.png','Resolution',300)
end


%Produces figures of model predicted errors

col_lim = 3*pi/4;

list_contours_1_2 = {B1_e, B2i_e, B2ii_e, ...
    B1_e, B2i_e, B2ii_e};
list_models_1_2 = {A_B1_APPROX, A_B2i_APPROX, A_B2ii_APPROX, ...
    A_B1_DIREC,  A_B2i_DIREC, A_B2ii_DIREC};
title_models_1_2 = ["B1 APPROX", "B2i APPROX", "B2ii APPROX", ...
    "B1 DIRECTIONAL", "B2i DIRECTIONAL", "B2ii DIRECTIONAL"];

list_contours3i = {B3i_e, B3i_e, B3i_e, B3i_e, B3i_e, B3i_e, B3i_e, B3i_e, B3i_e};
list_models3i = {A_B3i_CORRECT_T, A_B3i_CORRECT_R, A_B3i_CORRECT_Train, A_B3i_APPROX_T, A_B3i_APPROX_R, A_B3i_APPROX_Train,  ...
    A_B3i_DIREC_T, A_B3i_DIREC_R, A_B3i_DIREC_Train};
title_models_3i = ["B3i CORRECT: T", "B3i CORRECT: R", "B3i CORRECT: Tr", ...
    "B3i APPROX: T", "B3i APPROX: R", "B3i APPROX: Tr", ...
    "B3i DIRECTIONAL: T", "B3i DIRECTIONAL: R", "B3i DIRECTIONAL: Tr"];

list_models3ii = {A_B3ii_CORRECT_T, A_B3ii_CORRECT_R, A_B3ii_CORRECT_Train, A_B3ii_APPROX_T, A_B3ii_APPROX_R, A_B3ii_APPROX_Train, ...
    A_B3ii_DIREC_T A_B3ii_DIREC_R A_B3ii_DIREC_Train};
list_contours3ii = {B3ii_e, B3ii_e, B3ii_e, B3ii_e, B3ii_e, B3ii_e, B3ii_e, B3ii_e, B3ii_e};
title_models_3ii = ["B3ii CORRECT: T", "B3ii CORRECT: R", "B3ii CORRECT: Tr", ...
    "B3ii APPROX: T", "B3ii APPROX: R", "B3ii APPROX: Tr", ...
    "B3ii DIRECTIONAL: T", "B3ii DIRECTIONAL: R", "B3ii DIRECTIONAL: Tr"];

list_models4i = {A_B4ii_CORRECT_T, A_B4ii_CORRECT_R, A_B4ii_CORRECT_Train, A_B4ii_DIREC_T};
list_contours4i = {B4i_e, B4i_e, B4i_e, B4i_e};
title_models_4i = ["B4i CORRECT/APPROX: T", "B4i CORRECT/APPROX: R", ...
    "B4i CORRECT/APPROX: Tr", "B4i DIRECTIONAL"];

list_models4ii = {A_B4ii_CORRECT_T, A_B4ii_CORRECT_R, A_B4ii_CORRECT_Train, A_B4ii_DIREC_T};
list_contours4ii = {B4ii_e, B4ii_e, B4ii_e, B4ii_e};
title_models_4ii = ["B4ii CORRECT/APPROX: T", "B4ii CORRECT/APPROX: R", ...
    "B4ii CORRECT/APPROX: Tr", "B4ii DIRECTIONAL"];

B1_2_mod_fig = function_mod_fig(col_lim, 400, 250, 2, 3, list_contours_1_2, list_models_1_2, title_models_1_2, A_e, TRUE, x, y, limx, limy);

B3i_mod_fig = function_mod_fig(col_lim, 400, 400, 3, 3, list_contours3i, list_models3i, title_models_3i, A_e, TRUE, x, y, limx, limy);
B3ii_mod_fig = function_mod_fig(col_lim, 400, 400, 3, 3, list_contours3ii, list_models3ii, title_models_3ii, A_e, TRUE, x, y, limx, limy);

B4i_mod_fig = function_mod_fig(col_lim, 350, 350, 2, 2, list_contours4i, list_models4i, title_models_4i, A_e, TRUE, x, y, limx, limy);
B4ii_mod_fig = function_mod_fig(col_lim, 350, 350, 2, 2, list_contours4ii, list_models4ii, title_models_4ii, A_e, TRUE, x, y, limx, limy);

if model_figs_out
    exportgraphics(B1_2_mod_fig,'Write up/B1_2_mod_fig.png','Resolution',300)
    exportgraphics(B3i_mod_fig,'Write up/B3i_mod_fig.png','Resolution',300)
    exportgraphics(B3ii_mod_fig,'Write up/B3ii_mod_fig.png','Resolution',300)
    exportgraphics(B4i_mod_fig,'Write up/B4i_mod_fig.png','Resolution',300)
    exportgraphics(B4ii_mod_fig,'Write up/B4ii_mod_fig.png','Resolution',300)
end


