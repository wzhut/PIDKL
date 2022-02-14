clear all;
close all;


%load('res_PIGP.mat');
load('./res_PRGP.mat');
x = [xtr;xtest];
y = [ytr;y_test];
pred_mean = [pred_mean_tr;pred_mean];
pred_var = [pred_var_tr;pred_var];
upper = pred_mean + sqrt(pred_var);
lower = pred_mean - sqrt(pred_var);
hold on;
plot(x,y,'k','LineWidth',3);
plot(x,pred_mean,'r','LineWidth',3);
plot(x,lower,'r-.','LineWidth',3);
plot(x,upper,'r-.','LineWidth',3);


ylim([0,1.8]);

stdshade(pred_mean,sqrt(pred_var),0.1,'r',x,[])

[x,y] = meshgrid(0.1,0:0.1:1);

% figure(2)
plot(x(:),y(:),'g-.','LineWidth',3)
set(gca,'FontSize',45)
box on
grid on 
xlim([0,1])
ylim([0,1])
xticks([ 0.1, 0.5, 1])
yticks([0:0.5:1])
xlabel('t');
%ylabel('f(t)');



print('extrapo_prgp','-depsc')

% pred_var = pred_var';
% gtruth = y_test;
% plot(0.101:0.001:1.0, pred_mean, 'r');
% hold on;
% plot(0.101:0.001:1.0, gtruth, 'k');
% upper = pred_mean + sqrt(pred_var);
% lower = pred_mean - sqrt(pred_var);
% plot(0.101:0.001:1.0, upper, 'r');
% plot(0.101:0.001:1.0, lower, 'r');



