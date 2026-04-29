%% plot_eval_trajectory.m
% 功能：读取 Python 导出的评估轨迹 CSV 并在 MATLAB 中绘制轨迹图。
% 对 scripts/evaluate.py 导出的评估轨迹进行离线绘图。
%
% 用法：
%   1) 设置下面的 STAGE_DIR
%   2) 在 MATLAB 中运行本脚本
%
% 输出：
%   STAGE_DIR/figures/*.pdf 和 *.png

clear; clc; close all;

%% 配置
STAGE_DIR = 'data/eval_traj/demo_run/stage_1';  % <- 修改这里
PLOT_PER_EPISODE = true;
PLOT_COMBINED    = true;
SHOW_FIG         = true;

FIG_DIR = fullfile(STAGE_DIR, 'figures');
if ~exist(FIG_DIR, 'dir'), mkdir(FIG_DIR); end

files = dir(fullfile(STAGE_DIR, 'episode_*', 'trajectory.csv'));
if isempty(files)
    error('No trajectory.csv found under %s', STAGE_DIR);
end

allD = cell(numel(files),1);
allNames = cell(numel(files),1);

for i = 1:numel(files)
    csvPath = fullfile(files(i).folder, files(i).name);
    D = readtable(csvPath);
    [~, epName] = fileparts(files(i).folder);

    allD{i} = D;
    allNames{i} = epName;

    if PLOT_PER_EPISODE
        f = figure('Name', epName, 'Position', [100 100 1400 820]);

        % 3D 轨迹
        subplot(2,2,1); hold on; grid on;
        plot3(D.drone_x, D.drone_y, D.drone_z, '-', 'LineWidth', 1.6);
        plot3(D.platform_x, D.platform_y, D.platform_z, '-', 'LineWidth', 1.2);
        plot3(D.target_x, D.target_y, D.target_z, '--', 'LineWidth', 1.2);
        scatter3(D.drone_x(1), D.drone_y(1), D.drone_z(1), 36, 'filled');
        scatter3(D.drone_x(end), D.drone_y(end), D.drone_z(end), 40, 'x');

        hx = D.platform_hx(1); hy = D.platform_hy(1); hz = D.platform_hz(1);
        drawPlatformOutline(D.platform_x(1), D.platform_y(1), D.platform_z(1), hx, hy, hz, [0.8 0.2 0.2], 0.4);
        drawPlatformOutline(D.platform_x(end), D.platform_y(end), D.platform_z(end), hx, hy, hz, [0.8 0.2 0.2], 0.9);

        xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
        title('3D Trajectory');
        legend({'drone','platform','target','start','end'}, 'Location','best');
        axis equal;

        % 速度
        subplot(2,2,2); hold on; grid on;
        plot(D.sim_time, D.drone_vx, '-');
        plot(D.sim_time, D.drone_vy, '-');
        plot(D.sim_time, D.drone_vz, '-');
        plot(D.sim_time, D.speed_norm, '-', 'LineWidth', 1.8);
        xlabel('sim time (s)'); ylabel('m/s');
        title('Velocity');
        legend({'vx','vy','vz','|v|'}, 'Location','best');

        % 距离
        subplot(2,2,3); hold on; grid on;
        plot(D.sim_time, D.dist_3d, '-', 'LineWidth', 1.5);
        plot(D.sim_time, D.dist_xy, '-', 'LineWidth', 1.2);
        xlabel('sim time (s)'); ylabel('m');
        title('Distance to Target');
        legend({'dist_3d','dist_xy'}, 'Location','best');

        % xyz 位置
        subplot(2,2,4); hold on; grid on;
        plot(D.sim_time, D.drone_x, '-', 'LineWidth', 1.4);
        plot(D.sim_time, D.drone_y, '--', 'LineWidth', 1.4);
        plot(D.sim_time, D.drone_z, ':', 'LineWidth', 1.6);
        xlabel('sim time (s)'); ylabel('m');
        title('Drone Position');
        legend({'x','y','z'}, 'Location','best');

        sgtitle(sprintf('Trajectory Overview - %s', epName));

        saveas(f, fullfile(FIG_DIR, sprintf('%s_overview.png', epName)));
        saveas(f, fullfile(FIG_DIR, sprintf('%s_overview.pdf', epName)));
        if ~SHOW_FIG, close(f); end
    end
end

if PLOT_COMBINED
    f2 = figure('Name', 'Combined', 'Position', [120 120 1700 520]);

    subplot(1,3,1); hold on; grid on;
    C = lines(numel(allD));
    for i = 1:numel(allD)
        D = allD{i};
        plot3(D.drone_x, D.drone_y, D.drone_z, '-', 'Color', C(i,:), 'LineWidth', 1.2);
        plot3(D.platform_x, D.platform_y, D.platform_z, '-', 'Color', C(i,:), 'LineWidth', 0.8, 'Color', [C(i,:) 0.4]);
    end
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('3D Trajectory (Combined)');
    legend(allNames, 'Location', 'best');
    axis equal;

    subplot(1,3,2); hold on; grid on;
    for i = 1:numel(allD)
        D = allD{i};
        plot(D.sim_time, D.speed_norm, '-', 'Color', C(i,:), 'LineWidth', 1.2);
    end
    xlabel('sim time (s)'); ylabel('m/s');
    title('Speed Norm (Combined)');
    legend(allNames, 'Location', 'best');

    subplot(1,3,3); hold on; grid on;
    for i = 1:numel(allD)
        D = allD{i};
        plot(D.sim_time, D.drone_x, '-', 'Color', C(i,:), 'LineWidth', 1.1);
        plot(D.sim_time, D.drone_y, '--', 'Color', C(i,:), 'LineWidth', 1.1);
        plot(D.sim_time, D.drone_z, ':', 'Color', C(i,:), 'LineWidth', 1.3);
    end
    plot(nan, nan, 'k-', 'LineWidth', 1.1);
    plot(nan, nan, 'k--', 'LineWidth', 1.1);
    plot(nan, nan, 'k:', 'LineWidth', 1.3);
    xlabel('sim time (s)'); ylabel('m');
    title('Drone XYZ (Combined)');
    legend({'x','y','z'}, 'Location', 'best');

    sgtitle('Combined Trajectory Overview');
    saveas(f2, fullfile(FIG_DIR, 'combined_overview.png'));
    saveas(f2, fullfile(FIG_DIR, 'combined_overview.pdf'));
    if ~SHOW_FIG, close(f2); end
end

fprintf('Trajectory figures saved to: %s\n', FIG_DIR);

%% 局部函数
function drawPlatformOutline(cx, cy, cz, hx, hy, hz, color, alpha)
    zTop = cz + hz;
    xs = [cx-hx, cx+hx, cx+hx, cx-hx, cx-hx];
    ys = [cy-hy, cy-hy, cy+hy, cy+hy, cy-hy];
    zs = [zTop, zTop, zTop, zTop, zTop];
    plot3(xs, ys, zs, '-', 'Color', color, 'LineWidth', 1.4);
    if nargin >= 8
        h = plot3(xs, ys, zs, '-');
        h.Color = [color, alpha];
    end
end
