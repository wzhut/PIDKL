clear all;
close all;


% load('PIGP_extrap_0_res.mat');
load('diffuBC_1d_v1.mat');
load('GPR_SKL_res.mat');

h = figure;
hold on
x = reshape(X(:, 1), 48, 101);
t = reshape(X(:, 2), 48, 101);
te_pred_mean = reshape(te_pred_mean, 48, 101);
pcolor(t, x, te_pred_mean);

shading interp
colormap hot
caxis([1, 1.6]);
% colorbar('northoutside')
%     colorbar
%     legend(num2str(id(:)))
% caxis([0,0.1]);
ax = gca;
% ax.FontSize = 20;
%     ax.XLim = [1,length(nTr_lv2_list)];
% ax.XTick = [];
% ax.YTick = [];
% ylim([0, 2]);
% xlim([0, 1]);

% caxis([0,0.1]);
% xlabel('t')
% ylabel('x')
ax.FontSize = 45;

% box on 
% grid on

plot(xtr_t15(:, 2), xtr_t15(:, 1), 'g-');
% plot(1:10);
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(h,'SKL_diff.pdf','-dpdf','-r0')
% print(gcf, 'groundTruth.pdf', '-dpdf')






