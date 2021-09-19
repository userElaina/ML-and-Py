## Manual document
### 配置文件(pbc.conf)格式
参数名 ：参数值
### 参数说明
input--- 输入数据
NLabel--- 输入数据负样本类别名称
PLabel--- 输入数据正样本类别名称
OutputDir--- 输出文件路径
T-test-Output--- t检验特征保存文件名（作业4）
MaxFeature--- t检验输出排名前几的特征数量
Dp-Output--- 散点图保存文件名 （作业5）
subplot1--- 散点子图1的x轴特征排名
subplot2--- 散点子图2的x轴特征排名
subplot3--- 散点子图3的x轴特征排名
subplot4--- 散点子图4的x轴特征排名
H-Output1--- 柱形统计图1的保存文件名（作业6）
H-Output2--- 柱形统计图2的保存文件名（其他特征选择方法）
Hm-Outputx--- 热图x的保存文件名
svm-chosen-c--- SVM特征选择参数$C$
lasso-chosen-alpha--- Lasso特征选择参数$\alpha$
svm-feature-output--- SVM筛选特征的保存文件名
lasso-feature-output--- Lasso筛选特征的保存文件名
regularize--- 是否进行标准化
H1-statistic-output--- 柱形统计图1数据保存文件名
H2-statistic-output--- 柱形统计图2数据保存文件名

###输出文件说明
文件夹 *outputs0* 和 *outputs1* 分别是非标准化和标准化情况下的输出文件夹。
文件夹内部文件命名参考上述配置说明。

###运行方式及注意事项
程序运行时会自动打开 *pbc.conf* 并读取。
读取和存放都是使用相对路径（相对当前工作目录）。