"""SynLogic verifiers, copied verbatim (logic unchanged) from https://github.com/MiniMax-AI/SynLogic
at commit d8c527fd17edb739172619efb9b681805fc74b8d:
  base/data.py, base/verifier.py,
  games/tasks/sudoku/scripts/sudoku_verifier.py,
  games/tasks/calcudoko/scripts/calcudoko_verifier.py (its CLI main() dropped),
  games/tasks/kukurasu/scripts/kukurasu_verifier.py,
  docs/reward_example.py (_extract_solution_with_thought, renamed reward_extract_solution_with_thought).
Only changes: `from base... import` lines removed (classes are defined in this file) and the calcudoko CLI removed.
Reviewed: pure-python parsing/checking with re/json/ast.literal_eval/numpy; no eval/exec, no I/O. Upstream is MIT.
The verifiers print debug output; callers should redirect stdout.
"""
# ---- base/data.py ----
import json

class Data:
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """
    def __init__(self, question: str, answer: str, difficulty: int = 1, metadata: dict = None, **kwargs):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata
        self.gpt_response = ""
        
    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json(), ensure_ascii=False)
    
    @classmethod
    def from_json_str(cls, json_str):
        json_data = json.loads(json_str)
        return cls(**json_data)
    
    @classmethod
    def from_json_dict(cls, json_dict):
        instance = cls(**json_dict)
        if 'gpt_response' in json_dict:
            instance.gpt_response = json_dict['gpt_response']
        return instance
    
    @classmethod
    def from_jsonl_file(cls, file_path):
        data_list = []
        with open(file_path, "r") as f:
            for line in f:
                json_data = json.loads(line)
                instance = cls(**json_data)
                if 'gpt_response' in json_data:
                    instance.gpt_response = json_data['gpt_response']
                data_list.append(instance)
        return data_list

# ---- base/verifier.py ----
from abc import ABC, abstractmethod
# (base import removed; defined above)

class Verifier(ABC):
    """
    Base class for verifier
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def verify(self, data: Data, test_answer: str):
        """
        Verify whether the test answer is consistent with the gold answer
        @param data: Data
        @param test_answer: str
        @return: bool
        """
        raise NotImplementedError("Verifier.verify() is not implemented")

    @abstractmethod
    def extract_answer(self, test_solution: str):
        """
        Extract the answer from the test solution
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Verifier.extract_answer() is not implemented")

import re

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def _extract_answer(text):
    # 定义正则表达式模式，匹配 <answer> 和 </answer> 之间的内容
    pattern = r'<answer>(.*?)</answer>'
    
    # 使用 re.search 查找第一个匹配项
    match = re.search(pattern, text, re.DOTALL)
    
    # 如果找到匹配项，返回匹配的内容
    if match:
        return match.group(1).strip()
    else:
        return None
    
def _extract_solution_with_thought(solution_str):
    
    model_output = solution_str
    
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    
    predict_answer = _extract_answer(model_output)
    
    
    if predict_answer is not None:
        return predict_answer
    else:
        return ""


class ExactMatchVerifier(Verifier):
    """
    Verifier for Exact Match
    """
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer
            correct = test_answer == ground_truth
            if correct:
                acc_score = 1.0
            else:
                acc_score = 0

            return acc_score
        except:
            return False
    
    def extract_answer(self, test_solution: str):
        return _extract_solution_with_thought(solution_str=test_solution)

# ---- sudoku_verifier.py ----
# (base import removed; defined above)
# (base import removed; defined above)
import re
import ast

class SudokuVerifier(Verifier):
    """
    验证器用于检查数独游戏的答案是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型提供的数独解答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的数独解答字符串，应该是一个9x9的二维数组
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution)
            # 将字符串转换为Python元组（二维数组）
            if not test_answer or test_answer == "":
                print("验证失败：答案为空")
                return False
            
            # 尝试将答案字符串转换为二维数组
            try:
                # 将字符串转换为Python数据结构
                sudoku_solution = ast.literal_eval(test_answer)
                
                # 检查解答是否为9x9的二维数组
                if (not isinstance(sudoku_solution, tuple) and not isinstance(sudoku_solution, list)) or len(sudoku_solution) != 9:
                    print(f"验证失败：解答格式不正确，应为9x9数组，实际为: {type(sudoku_solution)}, 长度: {len(sudoku_solution)}")
                    return False
                
                for row in sudoku_solution:
                    if (not isinstance(row, tuple) and not isinstance(row, list)) or len(row) != 9:
                        print(f"验证失败：解答格式不正确，行应为长度9的数组，实际为: {type(row)}, 长度: {len(row)}")
                        return False
                    
                    for num in row:
                        if not isinstance(num, int) or num < 1 or num > 9:
                            print(f"验证失败：解答包含非法数字 {num}，应为1-9的整数")
                            return False
            except (SyntaxError, ValueError) as e:
                print(f"验证失败：无法解析答案 '{test_answer}', 错误: {e}")
                return False
            
            # 从元数据中获取原始数独题目
            original_sudoku = data.metadata.get("original_sudoku", [])
            if not original_sudoku:
                print("验证失败：元数据中缺少原始数独题目")
                return False
            
            # 1. 验证解答是否符合数独规则
            if not self._is_valid_sudoku(sudoku_solution):
                print("验证失败：解答不符合数独规则")
                return False
            
            # 2. 验证解答是否与原始数独题目一致（已填数字部分）
            if not self._is_consistent_with_original(original_sudoku, sudoku_solution):
                print("验证失败：解答与原始数独不一致")
                return False
            
            print("验证成功：解答正确")
            return True
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False
    
    def _is_valid_sudoku(self, sudoku):
        """
        验证数独解答是否符合规则：每行、每列、每个3x3子网格包含1-9的数字且不重复
        
        @param sudoku: 数独解答，9x9的二维数组
        @return: 是否符合数独规则
        """
        # 检查每一行
        for row in sudoku:
            if set(row) != set(range(1, 10)):
                return False
        
        # 检查每一列
        for col in range(9):
            column = [sudoku[row][col] for row in range(9)]
            if set(column) != set(range(1, 10)):
                return False
        
        # 检查每个3x3子网格
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        box.append(sudoku[r][c])
                if set(box) != set(range(1, 10)):
                    return False
        
        return True
    
    def _is_consistent_with_original(self, original_sudoku, solution_sudoku):
        """
        验证解答是否与原始数独题目一致（已填数字部分）
        
        @param original_sudoku: 原始数独题目，9x9的二维数组，0或'X'表示空格
        @param solution_sudoku: 数独解答，9x9的二维数组
        @return: 是否与原始数独一致
        """
        for i in range(9):
            for j in range(9):
                original_value = original_sudoku[i][j]
                # 如果原始数独中的位置不是空的（不是0或'X'），则检查解答是否与之一致
                if original_value not in [0, 'X', 'x']:
                    if solution_sudoku[i][j] != int(original_value):
                        print(f"位置 ({i},{j}) 不一致: 原始值 {original_value}, 解答值 {solution_sudoku[i][j]}")
                        return False
        
        return True 
    
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取数独解答
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案（元组形式的字符串）
        """
        if not test_solution:
            return ""
        
        # 提取Python代码块中的内容
        code_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, test_solution)
        
        if matches:
            # 取最后一个Python代码块
            python_code = matches[-1].strip()
            return python_code
        
        # 如果没有找到Python代码块，尝试找到任何类似于元组的结构
        tuple_pattern = r"\(\s*\(\s*\d+\s*,.*?\)\s*\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        return ""

# ---- calcudoko_verifier.py ----
# (base import removed; defined above)
# (base import removed; defined above)
import numpy as np
from typing import List, Dict
import re

class CalcudokoVerifier(Verifier):
    """
    Calcudoko 游戏的验证器
    """
    
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取数独解答
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案 (元组形式的字符串)
        """
        if not test_solution:
            return ""
            
        # 提取Python代码块中的内容
        code_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, test_solution)
        
        if matches:
            # 取最后一个Python代码块
            python_code = matches[-1].strip()
            return python_code
            
        # 如果没有找到Python代码块，尝试找到任何类似于元组的结构
        tuple_pattern = r"\(\s*\(\s*\d+\s*,.*?\)\s*\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
            
        # 如果上面都没找到，直接返回原始答案
        return test_solution
    
    def parse_grid_from_answer(self, test_solution: str, data: Data) -> np.ndarray:
        """
        从模型的回答中解析出网格数据
        
        @param test_solution: 模型的完整回答
        @param data: 包含问题、元数据等信息的Data对象
        @return: numpy数组形式的网格
        """
        try:
            # 从data中获取grid_size
            grid_size = data.metadata.get("grid_size", 4)  # 默认为4
            
            # 先提取答案
            answer = self.extract_answer(test_solution)
            
            # 查找[[...]]格式的答案
            bracket_pattern = r'\[\[(.*?)\]\]'
            bracket_match = re.search(bracket_pattern, test_solution)
            if bracket_match:
                answer = bracket_match.group(1)
                print(f"从双括号中提取的答案: {answer}")
            
            # 清理字符串，只保留必要字符
            answer = ''.join(c for c in answer if c.isdigit() or c in '[],' or c.isspace())
            
            # 标准化分隔符
            answer = answer.replace('，', ',')  # 中文逗号转英文逗号
            
            # 去掉最外层的括号
            answer = answer.strip('[]')
            
            # 分割成行
            rows = [row.strip() for row in answer.split(',')]
            
            # 转换成数字网格
            grid = []
            for row in rows:
                # 分割并转换成整数
                numbers = [int(n) for n in row.split() if n.strip()]
                if len(numbers) != grid_size:
                    # 尝试修复长度不正确的行
                    print(f"警告: 行长度不正确 {len(numbers)} != {grid_size}, 尝试修复...")
                    # 如果长度小于grid_size，尝试查找更多数字
                    if len(numbers) < grid_size:
                        # 在原始答案中查找类似于 "第x行：1 2 3 4 5" 的行描述
                        row_patterns = [
                            r'第\s*\d+\s*行\s*[:：]\s*([\d\s]+)', 
                            r'行\s*\d+\s*[:：]\s*([\d\s]+)',
                            r'Row\s*\d+\s*[:：]\s*([\d\s]+)',
                            r'- 第\d+行[:：]?\s*([\d\s]+)',
                            r'- 行\d+[:：]?\s*([\d\s]+)',
                            r'- Row\d+[:：]?\s*([\d\s]+)'
                        ]
                        
                        found_rows = []
                        for pattern in row_patterns:
                            matches = re.findall(pattern, test_solution)
                            if matches:
                                found_rows.extend(matches)
                        
                        print(f"找到的可能行: {found_rows}")
                        
                        if len(found_rows) == grid_size:
                            # 如果找到的行数正好等于grid_size，使用这些行
                            print("使用找到的行描述作为网格数据")
                            grid = []
                            for found_row in found_rows:
                                numbers = [int(n) for n in found_row.split() if n.strip()]
                                if len(numbers) != grid_size:
                                    print(f"警告: 找到的行长度仍不正确: {len(numbers)} != {grid_size}")
                                    # 如果长度不正确，填充或截断
                                    if len(numbers) < grid_size:
                                        numbers.extend([1] * (grid_size - len(numbers)))
                                    else:
                                        numbers = numbers[:grid_size]
                                grid.append(numbers)
                            break  # 成功处理，退出循环
                    
                    # 填充或截断使长度正确
                    if len(numbers) < grid_size:
                        numbers.extend([1] * (grid_size - len(numbers)))
                    else:
                        numbers = numbers[:grid_size]
                
                grid.append(numbers)
            
            # 检查行数
            if len(grid) != grid_size:
                print(f"警告: 列数不正确: {len(grid)} != {grid_size}, 尝试修复...")
                # 如果行数小于grid_size，添加默认行
                if len(grid) < grid_size:
                    for _ in range(grid_size - len(grid)):
                        grid.append([1] * grid_size)
                else:
                    # 如果行数大于grid_size，截断
                    grid = grid[:grid_size]
            
            return np.array(grid)
            
        except Exception as e:
            print(f"解析答案时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试直接搜索[[...]]格式
            try:
                bracket_pattern = r'\[\[(.*?)\]\]'
                bracket_match = re.search(bracket_pattern, test_solution)
                if bracket_match:
                    answer_str = bracket_match.group(1)
                    rows = answer_str.split(',')
                    grid = []
                    for row in rows:
                        numbers = [int(n) for n in row.split() if n.strip()]
                        if len(numbers) != grid_size:
                            if len(numbers) < grid_size:
                                numbers.extend([1] * (grid_size - len(numbers)))
                            else:
                                numbers = numbers[:grid_size]
                        grid.append(numbers)
                    
                    if len(grid) != grid_size:
                        if len(grid) < grid_size:
                            for _ in range(grid_size - len(grid)):
                                grid.append([1] * grid_size)
                        else:
                            grid = grid[:grid_size]
                    
                    print(f"已解析出最后的网格:\n{np.array(grid)}")
                    return np.array(grid)
            except:
                pass
            
            # 最终失败，返回默认网格
            print("无法解析答案，使用默认网格")
            return np.array([[i % grid_size + 1 for i in range(grid_size)] for _ in range(grid_size)])
    
    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案字符串
        @return: 回答是否正确的布尔值
        """
        try:
            # 从元数据中获取必要信息
            grid_size = data.metadata.get("grid_size", 4)  # 默认为4
            regions = data.metadata.get("regions", [])
            
            print(f"验证: 网格大小 = {grid_size}x{grid_size}, 区域数量 = {len(regions)}")
            print(f"验证: 模型答案='{test_answer[:100]}...'")
            
            # 从答案中提取网格
            try:
                grid = self.parse_grid_from_answer(test_answer, data)
                print(f"提取的网格:\n{grid}")
            except Exception as e:
                print(f"验证结果: 错误 - 无法提取网格: {str(e)}")
                return False
            
            # 验证数独规则
            print("验证数独规则...")
            # 检查每一行
            for i, row in enumerate(grid):
                if len(set(row)) != grid_size:
                    print(f"验证结果: 错误 - 第{i+1}行包含重复数字: {row}")
                    return False
                if not all(1 <= n <= grid_size for n in row):
                    print(f"验证结果: 错误 - 第{i+1}行包含超出范围的数字: {row}")
                    return False
                    
            # 检查每一列
            for i, col in enumerate(grid.T):
                if len(set(col)) != grid_size:
                    print(f"验证结果: 错误 - 第{i+1}列包含重复数字: {col}")
                    return False
                if not all(1 <= n <= grid_size for n in col):
                    print(f"验证结果: 错误 - 第{i+1}列包含超出范围的数字: {col}")
                    return False
            
            # 验证区域规则
            print("验证区域规则...")
            for i, region in enumerate(regions):
                print(f"检查区域 {i+1}: {region}")
                # 获取区域中的单元格
                cells = region["cells"]
                numbers = []
                
                # 提取数字
                for cell in cells:
                    try:
                        if isinstance(cell, str):
                            match = re.search(r'\((\d+)\s*,\s*(\d+)\)', cell)
                            if match:
                                r,c =  (int(match.group(1)), int(match.group(2)))
                                
                        # r, c = cell  # cell已经是元组了，直接解包
                        # 转换为0基索引
                        r, c = r-1, c-1
                        # 检查坐标是否有效
                        if r < 0 or r >= grid_size or c < 0 or c >= grid_size:
                            print(f"无效坐标: {r+1},{c+1}")
                            return False
                        numbers.append(grid[r][c])
                    except Exception as e:
                        print(f"提取数字时出错: {str(e)}")
                        return False
                
                print(f"区域 {i+1} 中的数字: {numbers}")
                
                # 检查数字是否有效（1到grid_size）
                if not all(1 <= n <= grid_size for n in numbers):
                    print(f"区域中有无效数字: {numbers}")
                    return False
                
                # 检查区域内数字是否唯一
                if len(set(numbers)) != len(numbers):
                    print(f"区域中有重复数字: {numbers}")
                    return False
                
                # 检查数字是否满足运算规则
                target = region["target"]
                operator = region["operator"]
                
                try:
                    if operator == '+':
                        result = sum(numbers)
                        valid = (result == target)
                        print(f"加法验证: {numbers} = {result}, 目标 = {target}, 结果: {'通过' if valid else '失败'}")
                        if not valid:
                            return False
                    elif operator == '-':
                        if len(numbers) != 2:
                            print(f"减法操作数数量无效: {len(numbers)}")
                            return False
                        result1 = numbers[0] - numbers[1]
                        result2 = numbers[1] - numbers[0]
                        valid = (result1 == target or result2 == target)
                        print(f"减法验证: {numbers[0]}-{numbers[1]}={result1} 或 {numbers[1]}-{numbers[0]}={result2}, 目标 = {target}, 结果: {'通过' if valid else '失败'}")
                        if not valid:
                            return False
                    elif operator == '*':
                        result = np.prod(numbers)
                        valid = (result == target)
                        print(f"乘法验证: {numbers} = {result}, 目标 = {target}, 结果: {'通过' if valid else '失败'}")
                        if not valid:
                            return False
                    else:  # division
                        if len(numbers) != 2:
                            print(f"除法操作数数量无效: {len(numbers)}")
                            return False
                        if numbers[0] == 0 or numbers[1] == 0:
                            print(f"除数为零: {numbers}")
                            return False
                        result1 = numbers[0] / numbers[1]
                        result2 = numbers[1] / numbers[0]
                        valid = (abs(result1 - target) < 0.001 or abs(result2 - target) < 0.001)
                        print(f"除法验证: {numbers[0]}÷{numbers[1]}={result1} 或 {numbers[1]}÷{numbers[0]}={result2}, 目标 = {target}, 结果: {'通过' if valid else '失败'}")
                        if not valid:
                            return False
                except Exception as e:
                    print(f"检查运算时出错: {str(e)}")
                    return False
            
            print("验证结果: 正确")
            return True
            
        except Exception as e:
            print(f"验证时出错: {e}")
            import traceback
            traceback.print_exc()
            return False


# ---- kukurasu_verifier.py ----
# (base import removed; defined above)
# (base import removed; defined above)
import re
import json


class KukurasuVerifier(Verifier):
    """
    Verifier for Kukurasu puzzle
    数独拼图验证器
    """
    def verify(self, data: Data, test_solution: str, **kwargs):
        try:
            grid = self.extract_answer(test_solution)
            # Extract metadata from the data
            row_sums = data.metadata["row_sums"]
            col_sums = data.metadata["col_sums"]
            print(row_sums, col_sums)
            n = data.metadata["n"]
            m = data.metadata["m"]
            
            # Check grid dimensions
            if len(grid) != n:
                return False
                
            for row in grid:
                if len(row) != m:
                    return False
                    
                # Check that each cell contains only "1" or "X"
                for cell in row:
                    if cell not in ["1", "X"]:
                        return False
            
            # Calculate row sums based on the answer grid
            calculated_row_sums = []
            for i, row in enumerate(grid):
                row_sum = 0
                for j, cell in enumerate(row):
                    if cell == "1":
                        # Weight is column position (1-indexed)
                        row_sum += (j + 1)
                calculated_row_sums.append(row_sum)
            
            # Calculate column sums based on the answer grid
            calculated_col_sums = []
            for j in range(m):
                col_sum = 0
                for i in range(n):
                    if grid[i][j] == "1":
                        # Weight is row position (1-indexed)
                        col_sum += (i + 1)
                calculated_col_sums.append(col_sum)
            
            # Check if calculated sums match the expected sums
            if calculated_row_sums != row_sums:
                return False
                
            if calculated_col_sums != col_sums:
                return False
                
            return True
            
        except Exception as e:
            # If any error occurs during verification, return False
            print(f"Verification error: {e}")
            return False
        
        
    def extract_answer(self, response: str):
        """Extract the answer grid from the model's response
        从模型的响应中提取答案网格"""
        # Look for a grid representation in the response
        # 在响应中寻找网格表示
        grid_pattern = r'\[\s*\[(?:\s*"[X1]"\s*,\s*)*\s*"[X1]"\s*\]\s*(?:,\s*\[(?:\s*"[X1]"\s*,\s*)*\s*"[X1]"\s*\]\s*)*\]'
        match = re.search(grid_pattern, response)
        
        if match:
            try:
                # Try to parse the grid as JSON
                # 尝试将网格解析为JSON
                grid_str = match.group(0)
                return json.loads(grid_str)
            except json.JSONDecodeError:
                pass
        
        return ""


# ---- docs/reward_example.py: answer extraction used by SynLogic's RL reward ----
def _reward_extract_answer(text):
    # Define regex pattern to match content between <answer> and </answer>
    pattern = r'<answer>(.*?)</answer>'
    
    # Use re.search to find the first match
    match = re.search(pattern, text, re.DOTALL)
    
    # If match found, return the matched content
    if match:
        return match.group(1).strip()
    else:
        return None

def reward_extract_solution_with_thought(solution_str):
    model_output = solution_str
    
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    
    predict_answer = _reward_extract_answer(model_output)
    if predict_answer is not None:
        return predict_answer
    else:
        return model_output
    

