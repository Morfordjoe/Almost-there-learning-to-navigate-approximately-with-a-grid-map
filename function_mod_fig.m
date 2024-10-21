function fig_out = function_mod_fig(col_lm, wd, ht, rws, cols, list_cont, list_md, list_titles_in, A_e_in, TRUE_in, x_in, y_in, limx_in, limy_in)
    
    %Produces multi-panel figure of model predictions

    figure('Position', [1, 1, wd, ht])
    n_mod = rws*cols;
    fig_out = tiledlayout(rws, cols, 'TileSpacing','Compact','Padding','Compact');


    %Changes scaling of color gradient
    c = parula;
    c1 = c(repmat(1:(end/5),5,1),:);
    c2 = c(repmat(round(end/5):(2*end/5),3,1),:);
    c3 = c(repmat(round(2*end/5):(3*end/5),1,1),:);
    c4 = c(repmat(round(3*end/5):(4*end/5),3,1),:);
    c5 = c(repmat(round(4*end/5):(5*end/5),5,1),:);
    c_full = [c1; c2; c3; c4; c5];

    for i = 1:n_mod
        nexttile
        fcontour_e = list_cont{i};
        col_vec = mod(list_md{i} - TRUE_in, 2*pi);
        col_vec(col_vec > pi) = col_vec(col_vec > pi) - 2*pi;
        col_vec(abs(col_vec) < 0.00001) = 0;
        scatter(x_in(isfinite(col_vec)),y_in(isfinite(col_vec)),25, col_vec(isfinite(col_vec)),'filled')
        axis equal
        title(list_titles_in(i))
        xlim([-limx_in-0.5, limx_in+0.5])
        ylim([-limy_in-0.5, limy_in+0.5])
        colormap(c_full);
        caxis([-col_lm, col_lm])
        hold on
        fcontour(A_e_in, [-limx_in-0.51, limx_in+0.51, -limy_in-0.51, limy_in+0.51], 'LineWidth',0.5, 'LineColor', 'black', 'LevelList', [-10:-1, 1:10])
        fcontour(fcontour_e, [-limx_in-0.51, limx_in+0.51, -limy_in-0.51, limy_in+0.51], '--', 'LineWidth',0.5, 'LineColor', 'red', 'LevelList', [-10:-1, 1:10])
        plot(0, 0, '.', 'Color', 'red', 'MarkerSize', 12)
        fcontour(A_e_in, [-limx_in-0.51, limx_in+0.51, -limy_in-0.51, limy_in+0.51], 'LineWidth',1.5, 'LineColor', 'black', 'LevelList', [0])
        fcontour(fcontour_e, [-limx_in-0.51, limx_in+0.51, -limy_in-0.51, limy_in+0.51], '--', 'LineWidth',1.5, 'LineColor', 'red', 'LevelList', [0])
        hold off
    
    end
    colorbar()
end
