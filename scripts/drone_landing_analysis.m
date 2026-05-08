%% drone_landing_analysis.m
% 功能：读取训练 CSV 数据并在 MATLAB 中分析奖励、成功率和误差指标。
% 无人机降落 RL 训练后分析。
%
% 读取训练期间导出的 CSV 文件，并复现 Python 绘图。
% 需要：Statistics and Machine Learning Toolbox（用于 violin plot 替代方案）
%
% 用法：
%   1. 将 CSV_DIR 设置为包含 episode_log.csv 等文件的目录。
%   2. 用 Ctrl+Enter 分段运行，或直接运行整个脚本。
%
% 输出：图像以 PDF 保存到 CSV_DIR/figures_matlab/

clear; clc; close all;

%% ── 配置 ─────────────────────────────────────────────────────────────────
CSV_DIR  = 'output/data/csv/drone_landing';   % ← 修改该路径
OUT_DIR  = fullfile(CSV_DIR, 'figures_matlab');
MA_WIN   = 100;           % 移动平均窗口（episode）
SR_WIN   = 50;            % 成功率移动平均窗口
SHOW_FIG = true;          % 设为 false 可关闭显示

STAGE_COLORS = [
    0.298 0.447 0.690;
    0.333 0.659 0.408;
    0.769 0.306 0.322;
    0.867 0.518 0.322;
];

STAGE_LABELS = {"Stage 1", "Stage 2", "Stage 3", "Stage 4"};

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

%% ── 加载数据 ─────────────────────────────────────────────────────────────
fprintf('Loading data from %s\n', CSV_DIR);

ep_path = fullfile(CSV_DIR, 'episode_log.csv');
rw_path = fullfile(CSV_DIR, 'reward_log.csv');
tr_path = fullfile(CSV_DIR, 'training_log.csv');

if ~isfile(ep_path)
    error('episode_log.csv not found in %s', CSV_DIR);
end

EP = readtable(ep_path);
fprintf('  episode_log: %d rows\n', height(EP));

RW = [];
if isfile(rw_path)
    RW = readtable(rw_path);
    fprintf('  reward_log : %d rows\n', height(RW));
else
    warning('reward_log.csv not found — some plots will be skipped.');
end

TR = [];
if isfile(tr_path)
    TR = readtable(tr_path);
    fprintf('  training_log: %d rows\n', height(TR));
else
    warning('training_log.csv not found — loss plots will be skipped.');
end

%% ── 图 1：Episode Return 曲线 ────────────────────────────────────────────
fig1 = figure('Name', 'Reward Curve', 'Position', [100 100 1000 400]);
hold on; grid on;

for s = 1:4
    mask = EP.stage == s;
    if ~any(mask), continue; end
    ts  = EP.timestep(mask);
    rwd = EP.reward(mask);
    ma  = movmean(rwd, MA_WIN);
    scatter(ts, rwd, 2, STAGE_COLORS(s,:), 'filled', ...
            'MarkerFaceAlpha', 0.12, 'HandleVisibility', 'off');
    plot(ts, ma, '-', 'Color', STAGE_COLORS(s,:), 'LineWidth', 2, ...
         'DisplayName', STAGE_LABELS{s});
end

xlabel('Timestep'); ylabel('Episode Return');
title(sprintf('Training Reward Curve (Smoothed, MA-%d)', MA_WIN));
legend('Location', 'southeast');
ax = gca; ax.Box = false; ax.XAxis.Exponent = 6;
saveas(fig1, fullfile(OUT_DIR, 'reward_curve.pdf'));
if ~SHOW_FIG, close(fig1); end

%% ── 图 2：成功率 ─────────────────────────────────────────────────────────
fig2 = figure('Name', 'Success Rate', 'Position', [100 100 1000 400]);
hold on; grid on;
yline(0.8, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, ...
      'DisplayName', 'Advance threshold (80%)');

for s = 1:4
    mask = EP.stage == s;
    if ~any(mask), continue; end
    ts  = EP.timestep(mask);
    sr  = movmean(double(EP.success(mask)), SR_WIN);
    plot(ts, sr, '-', 'Color', STAGE_COLORS(s,:), 'LineWidth', 2, ...
         'DisplayName', STAGE_LABELS{s});
end

xlabel('Timestep'); ylabel('Success Rate');
ylim([-0.05 1.05]); title(sprintf('Rolling Success Rate (MA-%d)', SR_WIN));
legend('Location', 'northwest');
ax = gca; ax.Box = false; ax.XAxis.Exponent = 6;
saveas(fig2, fullfile(OUT_DIR, 'success_rate.pdf'));
if ~SHOW_FIG, close(fig2); end

%% ── 图 3：Episode 长度 ───────────────────────────────────────────────────
fig3 = figure('Name', 'Episode Length', 'Position', [100 100 1000 400]);
hold on; grid on;

for s = 1:4
    mask = EP.stage == s;
    if ~any(mask), continue; end
    ts  = EP.timestep(mask);
    ma  = movmean(EP.length(mask), MA_WIN);
    plot(ts, ma, '-', 'Color', STAGE_COLORS(s,:), 'LineWidth', 2, ...
         'DisplayName', STAGE_LABELS{s});
end

xlabel('Timestep'); ylabel('Episode Length (steps)');
title(sprintf('Episode Length over Training (MA-%d)', MA_WIN));
legend('Location', 'northeast');
ax = gca; ax.Box = false; ax.XAxis.Exponent = 6;
saveas(fig3, fullfile(OUT_DIR, 'episode_length.pdf'));
if ~SHOW_FIG, close(fig3); end

%% ── 图 4：奖励分项趋势 ───────────────────────────────────────────────────
if ~isempty(RW)
    rw_cols = {'reward_dist','reward_smooth','reward_alive', ...
               'reward_vel_match','reward_yaw','reward_approach'};
    rw_names = {'Distance','Smoothness','Alive','Vel Match','Yaw','Approach'};

    fig4 = figure('Name', 'Reward Components', ...
                  'Position', [100 100 1400 700]);
    MW = 2000;  % 奖励移动平均窗口（step，不是 episode）

    for i = 1:6
        subplot(2, 3, i); hold on; grid on;
        col = rw_cols{i};
        if ismember(col, RW.Properties.VariableNames)
            ma = movmean(RW.(col), MW);
            for s = 1:4
                mask = RW.stage == s;
                if any(mask)
                    plot(RW.timestep(mask), ma(mask), '.', ...
                         'Color', STAGE_COLORS(s,:), 'MarkerSize', 1);
                end
            end
        end
        title(rw_names{i}); xlabel('Timestep'); ylabel('Value');
        ax = gca; ax.Box = false;
    end
    sgtitle(sprintf('Reward Component Trends (MA-%d)', MW), 'FontSize', 14);
    saveas(fig4, fullfile(OUT_DIR, 'reward_components.pdf'));
    if ~SHOW_FIG, close(fig4); end
end

%% ── 图 5：训练 Loss 曲线 ─────────────────────────────────────────────────
if ~isempty(TR)
    fig5 = figure('Name', 'Training Loss', 'Position', [100 100 1200 400]);

    subplot(1,3,1); hold on; grid on;
    plot(TR.timestep, TR.actor_loss, '-', 'Color', STAGE_COLORS(1,:), 'LineWidth', 1.5);
    title('Actor Loss'); xlabel('Timestep'); ylabel('Loss');
    gca().Box = false;

    subplot(1,3,2); hold on; grid on;
    plot(TR.timestep, TR.critic_loss, '-', 'Color', STAGE_COLORS(3,:), 'LineWidth', 1.5);
    title('Critic Loss'); xlabel('Timestep'); ylabel('Loss');
    gca().Box = false;

    subplot(1,3,3); hold on; grid on;
    plot(TR.timestep, TR.ent_coef, '-', 'Color', STAGE_COLORS(2,:), 'LineWidth', 1.5);
    title('Entropy Coefficient'); xlabel('Timestep'); ylabel('Value');
    gca().Box = false;

    sgtitle('SAC Training Metrics', 'FontSize', 14);
    saveas(fig5, fullfile(OUT_DIR, 'training_loss.pdf'));
    if ~SHOW_FIG, close(fig5); end
end

%% ── 图 6：各阶段距离分布（Box Plot）────────────────────────────────────
if ~isempty(RW) && ismember('metric_dist', RW.Properties.VariableNames)
    fig6 = figure('Name', 'Distance Distribution', 'Position', [100 100 700 500]);
    hold on; grid on;

    stage_data = cell(1,4);
    for s = 1:4
        mask = RW.stage == s;
        if any(mask)
            stage_data{s} = RW.metric_dist(mask);
        else
            stage_data{s} = [];
        end
    end

    % 为 boxplot 构造分组向量
    vals   = vertcat(stage_data{:});
    groups = vertcat( repmat(1, length(stage_data{1}), 1), ...
                      repmat(2, length(stage_data{2}), 1), ...
                      repmat(3, length(stage_data{3}), 1), ...
                      repmat(4, length(stage_data{4}), 1) );

    boxplot(vals, groups, 'Labels', {'S1','S2','S3','S4'}, ...
            'Colors', STAGE_COLORS, 'Symbol', '.', 'OutlierSize', 3);
    xlabel('Curriculum Stage'); ylabel('Distance to Target (m)');
    title('Distance-to-Target Distribution by Stage');
    ax = gca; ax.Box = false;
    saveas(fig6, fullfile(OUT_DIR, 'dist_distribution.pdf'));
    if ~SHOW_FIG, close(fig6); end
end

%% ── 各阶段汇总统计 ──────────────────────────────────────────────────────
fprintf('\n%s\n', repmat('=',1,60));
fprintf('  Per-Stage Summary Statistics\n');
fprintf('%s\n', repmat('=',1,60));
fprintf('  %-6s  %8s  %8s  %8s  %8s  %8s\n', ...
        'Stage','Episodes','Succ%','MeanR','StdR','MeanLen');
fprintf('  %s\n', repmat('-',1,56));

for s = 1:4
    mask = EP.stage == s;
    if ~any(mask), continue; end
    n   = sum(mask);
    sr  = mean(EP.success(mask)) * 100;
    mr  = mean(EP.reward(mask));
    str = std(EP.reward(mask));
    ml  = mean(EP.length(mask));
    fprintf('  Stage %d  %8d  %7.1f%%  %+8.2f  %8.2f  %8.1f\n', ...
            s, n, sr, mr, str, ml);
end
fprintf('%s\n', repmat('=',1,60));

fprintf('\nAll figures saved to: %s\n', OUT_DIR);
