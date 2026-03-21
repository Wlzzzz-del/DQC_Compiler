import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件，假设没有标题行，第一列是时间，第三列是Loss
# 注意：根据文件内容，该文件只有两列，这里假设第二列是Loss（可能用户指第二列）
data = pd.read_csv('c:/project/DQC_Compiler/running_result/run_on_10Q_10G_with_tofoli02doneTime.csv', header=None, names=['Time', 'Col2', 'Loss'], skiprows=1618)

# 如果第三列不存在，这会出错；如果文件确实有三列，请确认
# 绘制曲线图
plt.plot(data['Time'], data['Col2'])
plt.xlabel('时间')
plt.ylabel('Loss')
plt.title('Loss vs 时间')
plt.show()