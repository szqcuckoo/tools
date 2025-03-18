// 模型字符集包含英文数字与标点, 我们忽略所有非数字字符将其视为0
const ocr_character = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
    '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
];

let session, rawImage = new Image(), boxes = [];
const cells = []; // 保存表格单元格的二维数组
const imageElm = document.getElementById('inputImage');
const canvasElm = document.getElementById('canvasImage');
const statusElm = document.getElementById('boardStatus');
window.cells = cells;
window.boxes = boxes;

for (let i = 0; i < 9; i++) {
    const row = document.getElementById('sudokuBoard').insertRow();
    cells.push([]);
    for (let j = 0; j < 9; j++) {
        const cell = row.insertCell();
        cell.contentEditable = false;
        cells[i].push(cell);
    }
}

class recPostprocess {
    constructor(preds) {
        this.ocr_character = ocr_character;
        this.preds_idx = [];
        const pred_len = 97;
        for (let i = 0; i < preds.length; i += pred_len) {
            const tmpArr = preds.slice(i, i + pred_len);
            const tmpMax = Math.max(...tmpArr);
            const tmpIdx = tmpArr.indexOf(tmpMax);
            this.preds_idx.push(tmpIdx);
        }
    }

    decode(text_index, is_remove_duplicate = false) {
        const ignored_tokens = this.get_ignored_tokens();
        const char_list = [];
        for (let idx = 0; idx < text_index.length; idx++) {
            if (ignored_tokens.includes(text_index[idx])) {
                continue;
            }
            if (is_remove_duplicate) {
                if (idx > 0 && text_index[idx - 1] === text_index[idx]) {
                    continue;
                }
            }
            char_list.push(this.ocr_character[text_index[idx] - 1]);
        }
        return char_list;
    }

    get_ignored_tokens() { return [0]; }

    outputResult() {
        return this.decode(this.preds_idx, true);
    }
}

const recognize = async function (img) {
    const tensor = await ort.Tensor.fromImage(img, { tensorFormat: "RGB", norm: { bias: -127.5, mean: 127.5 } });
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    const output = (await session.run(feeds))[session.outputNames[0]];
    const results = new recPostprocess(output.data);
    tensor.dispose();
    output.dispose();

    return results.outputResult();
}

async function init() {
    ort.env.webgl.pack = true;
    ort.env.logLevel = 'verbose';
    ort.env.trace = true;
    ort.env.debug = true;
    ort.InferenceSession.create(window.modelPath)
        .then((ret) => {
            session = ret;
            document.getElementById('onnxStatus').innerHTML = 'PaddleOCR 加载完成';
            // 隐藏模型上传区域
            const modelLoaderElement = document.querySelector('.model-loader');
            if (modelLoaderElement) {
                modelLoaderElement.style.display = 'none';
            }
        })
        .catch((err) => {
            document.getElementById('onnxStatus').innerHTML = 'PaddleOCR 加载失败,请刷新页面重试';
            // 显示模型上传区域
            const modelLoaderElement = document.querySelector('.model-loader');
            if (modelLoaderElement) {
                modelLoaderElement.style.display = 'flex';
            }
        });
}

// 尝试从assets文件夹加载默认的paddle.onnx模型
function loadDefaultModel() {
    // 更新状态显示
    const statusElement = document.getElementById('onnxStatus');
    const modelLoaderElement = document.querySelector('.model-loader');
    
    // 默认隐藏模型上传区域，等待远程加载结果
    if (modelLoaderElement) {
        modelLoaderElement.style.display = 'none';
    }
    
    if (statusElement) {
        statusElement.innerHTML = '正在尝试加载默认OCR模型...';
    }
    
    fetch('./assets/paddle.onnx')
        .then(response => {
            if (!response.ok) {
                console.warn('无法从assets文件夹加载默认OCR模型，请手动上传模型文件');
                if (statusElement) {
                    statusElement.innerHTML = '请上传OCR模型文件(.onnx)';
                }
                // 显示模型上传区域
                if (modelLoaderElement) {
                    modelLoaderElement.style.display = 'flex';
                }
                return null;
            }
            return response.blob();
        })
        .then(blob => {
            if (blob) {
                window.modelPath = URL.createObjectURL(blob);
                console.log('已从assets文件夹加载默认OCR模型');
                // 成功加载远程模型，隐藏模型上传区域
                if (modelLoaderElement) {
                    modelLoaderElement.style.display = 'none';
                }
            }
        })
        .catch(error => {
            console.error('加载默认OCR模型时出错:', error);
            if (statusElement) {
                statusElement.innerHTML = '加载默认模型失败，请手动上传模型文件';
            }
            // 显示模型上传区域
            if (modelLoaderElement) {
                modelLoaderElement.style.display = 'flex';
            }
        });
}

// 页面加载完成后尝试加载默认模型
document.addEventListener('DOMContentLoaded', function() {
    // 只有当没有模型路径时才尝试加载默认模型
    if (!window.modelPath) {
        loadDefaultModel();
    }
    
    // 确保开始检查OCR加载状态
    if (!window.checkOCRInterval) {
        window.checkOCRInterval = setInterval(function() {
            if (window.ort) {
                clearInterval(window.checkOCRInterval);
                checkPaddleOCRLoaded();
            }
        }, 100);
    }
});

function checkPaddleOCRLoaded() {
    if ((window.ort) && (window.modelPath)) {
        init();
    } else if (window.ort && !window.modelPath) {
        // 如果onnxruntime已加载但没有模型路径，尝试加载默认模型
        loadDefaultModel();
        // 短暂延迟后再次检查
        setTimeout(checkPaddleOCRLoaded, 500);
    } else {
        // 等待一段时间后再次检查
        setTimeout(checkPaddleOCRLoaded, 100);
    }
};

function resetWorkspace() {
    imageElm.src = '';
    rawImage.src = '';
    const statusElm = document.getElementById('boardStatus');
    statusElm.innerHTML = '';
    statusElm.style.backgroundColor = '';
    statusElm.style.color = '';
    
    // 清空整个画布
    const ctx = canvasElm.getContext('2d');
    ctx.clearRect(0, 0, canvasElm.width, canvasElm.height);
    boxes = [];

    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            cells[i][j].style.backgroundColor = '';
            cells[i][j].style.color = '';
            cells[i][j].style.fontWeight = '';
            cells[i][j].textContent = '';
            cells[i][j].contentEditable = true;
        }
    }
};

function isValidSudoku(cells) {
    const rows = new Array(9).fill(0).map(() => new Array(9).fill({}));
    const columns = new Array(9).fill(0).map(() => new Array(9).fill({}));
    const subboxes = new Array(3).fill(0).map(() => new Array(3).fill(0).map(() => new Array(9).fill({})));
    const duplicated = new Array();
    let valid = true;
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = cells[i][j];
            const c = cell.textContent;
            cell.style.backgroundColor = (cell.contentEditable == 'true') ? null : '#edf2f7';
            // 多字符 非数字字符
            if ((c.length > 1) || (c.charCodeAt() < 49) || (c.charCodeAt() > 57)) {
                valid = false;
                duplicated.push({ x: i, y: j });
                continue
            }
            if (c !== '') {
                const index = c.charCodeAt() - '0'.charCodeAt() - 1;
                const ri = rows[i][index];
                const ci = columns[j][index];
                const bi = subboxes[Math.floor(i / 3)][Math.floor(j / 3)][index];
                if (ri.x >= 0) {
                    duplicated.push(ri);
                }
                if (ci.x >= 0) {
                    duplicated.push(ci);
                }
                if (bi.x >= 0) {
                    duplicated.push(bi);
                }
                if ((ri.x >= 0) || (ci.x >= 0) || (bi.x >= 0)) {
                    duplicated.push({ x: i, y: j });
                    valid = false;
                }
                rows[i][index] = { x: i, y: j };
                columns[j][index] = { x: i, y: j };
                subboxes[Math.floor(i / 3)][Math.floor(j / 3)][index] = { x: i, y: j };
            } else {
                valid = false;
            }
        }
    }
    duplicated.forEach((e) => {
        cells[e.x][e.y].style.backgroundColor = 'rgba(231, 76, 60, 0.3)'; // 更柔和的红色
    });
    return valid;
};

function preprocess(mat) {
    const dstMat = new cv.Mat();
    cv.cvtColor(mat, dstMat, cv.COLOR_GRAY2BGRA, 4);
    const dsize = new cv.Size(Math.floor(mat.cols / mat.rows * 48), 48);
    cv.resize(dstMat, dstMat, dsize, 0, 0, cv.INTER_LINEAR);

    const imgData = new ImageData(new Uint8ClampedArray(dstMat.data), dstMat.cols, dstMat.rows);
    dstMat.delete();
    return imgData;
}

function cleanContours(contours) {
    for (let i = 0; i < contours.size(); ++i) {
        const mat = contours.get(i);
        mat.delete(); // 释放cv.Mat对象占用的内存
    }
    contours.delete();
}

function findLines(img) {
    const meanHeight = Math.ceil(img.rows / 9);
    const meanWidth = Math.ceil(img.cols / 9);
    // 去除网格线
    let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.ceil(img.rows / 9), 1));
    const tmpImg = new cv.Mat();
    // 查找横线
    const hContours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.morphologyEx(img, tmpImg, cv.MORPH_OPEN, kernel);
    cv.findContours(tmpImg, hContours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // 查找竖线
    const vContours = new cv.MatVector();
    tmpImg.setTo(new cv.Scalar(0, 0, 0));
    kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(1, Math.ceil(img.rows / 9)));
    cv.morphologyEx(img, tmpImg, cv.MORPH_OPEN, kernel);
    cv.findContours(tmpImg, vContours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    hierarchy.delete();
    tmpImg.delete();

    const hLines = [];
    const vLines = [];
    for (let i = 0; i < hContours.size(); i++) {
        const rect = cv.boundingRect(hContours.get(i));
        if ((hContours.size() > 10) && (hLines.length > 0) && (Math.abs(hLines[hLines.length - 1].y - rect.y) < (meanHeight / 3))) {
            continue
        }
        hLines.push(rect);
    }
    for (let i = 0; i < vContours.size(); i++) {
        const rect = cv.boundingRect(vContours.get(i));
        if ((vLines.length > 0) && (vContours.size() > 10) && (Math.abs(vLines[vLines.length - 1].x - rect.x) < (meanWidth / 3))) {
            continue
        }
        vLines.push(rect);
    }

    vLines.reverse();
    hLines.reverse();

    const color = new cv.Scalar(0, 0, 0);
    cv.drawContours(img, vContours, -1, color, -1, cv.LINE_8);
    cv.drawContours(img, vContours, -1, color, 2, cv.LINE_8);
    cv.drawContours(img, hContours, -1, color, -1, cv.LINE_8);
    cv.drawContours(img, hContours, -1, color, 2, cv.LINE_8);

    cleanContours(hContours);
    cleanContours(vContours);

    return { h: hLines, v: vLines };
}

function findBoxes(lines, xOffset = 0, yOffset = 0) {
    const boxes = [];
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const left = lines.v[j], right = lines.v[j + 1], top = lines.h[i], btm = lines.h[i + 1];
            const x = left.x + left.width, y = top.y + top.height
            const w = right.x - x, h = btm.y - y;
            boxes.push({ x: x + xOffset, y: y + yOffset, width: w, height: h, isEmpty: false });
        }
    }
    return boxes;
}

async function recognizeByRow(img, lines) {
    const board = [];
    const meanHeight = Math.ceil(img.rows / 9);
    for (let i = 0; i < 9; i++) {
        const y = lines.h[i].y + lines.h[i].height;
        const row = img.roi({ x: lines.v[0].width, y: y, width: lines.v[9].x, height: Math.min(meanHeight, img.rows - y) });
        const rowNums = await recognize(preprocess(row));
        row.delete();
        if (rowNums.length !== 9) {
            return { full: false };
        }
        board.push(rowNums);
    }
    return { board: board, full: true };
}

async function recognizeByBoxes(img, boxes) {
    const board = [];
    for (let i = 0; i < 9; i++) {
        board.push(Array(9).fill('0'));
    }
    let full = true;
    for (let i = 0; i < boxes.length; ++i) {
        const x = Math.floor(i / 9);
        const y = i % 9;

        const box = img.roi(boxes[i]);
        if (cv.countNonZero(box) > 3) {
            const digits = await recognize(preprocess(box));
            board[x][y] = digits[0];
        } else {
            boxes[i].isEmpty = true;
            full = false;
        }
        box.delete();
    }
    return { board: board, full: full };
}

function fillIn(board, cells) {
    for (let i = 0; i < board.length; i++) {
        const row = board[i];
        for (let j = 0; j < row.length; j++) {
            if (board[i][j] !== '0') {
                cells[i][j].textContent = board[i][j];
                cells[i][j].contentEditable = false;
                cells[i][j].style.backgroundColor = '#edf2f7';
                cells[i][j].style.fontWeight = '600';
                cells[i][j].style.color = '#2c3e50';
            } else {
                cells[i][j].contentEditable = true;
                cells[i][j].style.backgroundColor = 'white';
                cells[i][j].style.color = '#3498db';
            }
        }
    }
}

window.genOutputPic = function () {
    const ctx = canvasElm.getContext('2d');

    canvasElm.width = rawImage.width;
    canvasElm.height = rawImage.height;
    // 在canvas上绘制内容
    ctx.drawImage(rawImage, 0, 0);
    ctx.fillStyle = '#ff0000';
    for (let k = 0; k < boxes.length; k++) {
        const i = Math.floor(k / 9), j = k % 9;
        const box = boxes[k];
        if (box.isEmpty) {
            ctx.font = 'bold ' + Math.ceil(box.height * 0.78) + 'px simsun';
            ctx.fillText(cells[i][j].textContent, box.x + Math.floor(box.width / 3), box.y + Math.ceil(box.height * 0.75));
        }
    }
    // 将canvas内容以图像URL的形式导出
    const dataURL = canvasElm.toDataURL();
    imageElm.src = dataURL;
};

window.checkAnswer = function () {
    const statusElm = document.getElementById('boardStatus');
    if (isValidSudoku(cells)) {
        statusElm.innerHTML = "答案正确";
        statusElm.style.backgroundColor = "rgba(46, 204, 113, 0.2)"; // 绿色背景
        statusElm.style.color = "#27ae60";
    } else {
        statusElm.innerHTML = "答案不正确";
        statusElm.style.backgroundColor = "rgba(231, 76, 60, 0.2)"; // 红色背景
        statusElm.style.color = "#c0392b";
    }
}

window.solveSudoku = function () {
    // 从界面读取当前数独状态
    const board = [];
    for (let i = 0; i < 9; i++) {
        board.push(Array(9).fill(0));
        for (let j = 0; j < 9; j++) {
            const cellContent = cells[i][j].textContent;
            board[i][j] = cellContent === "" ? 0 : parseInt(cellContent);
        }
    }
    
    const statusElm = document.getElementById('boardStatus');
    
    // 检查当前数独是否有效
    if (!isValidForSolving(board)) {
        statusElm.innerHTML = "当前数独状态无效，无法求解";
        statusElm.style.backgroundColor = "rgba(231, 76, 60, 0.2)"; // 红色背景
        statusElm.style.color = "#c0392b";
        return;
    }
    
    // 使用回溯算法解决数独
    if (solve(board)) {
        // 将解决方案填充到表格中
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                if (cells[i][j].textContent === "") {
                    cells[i][j].textContent = board[i][j].toString();
                    cells[i][j].style.backgroundColor = "rgba(46, 204, 113, 0.2)"; // 柔和的绿色
                }
            }
        }
        statusElm.innerHTML = "数独已成功解答";
        statusElm.style.backgroundColor = "rgba(46, 204, 113, 0.2)"; // 绿色背景
        statusElm.style.color = "#27ae60";
    } else {
        statusElm.innerHTML = "无法解答当前数独";
        statusElm.style.backgroundColor = "rgba(231, 76, 60, 0.2)"; // 红色背景
        statusElm.style.color = "#c0392b";
    }
}

// 检查数独状态是否有效（用于求解前的检查）
function isValidForSolving(board) {
    // 检查行
    for (let i = 0; i < 9; i++) {
        const row = new Set();
        for (let j = 0; j < 9; j++) {
            if (board[i][j] !== 0) {
                if (row.has(board[i][j])) return false;
                row.add(board[i][j]);
            }
        }
    }
    
    // 检查列
    for (let j = 0; j < 9; j++) {
        const col = new Set();
        for (let i = 0; i < 9; i++) {
            if (board[i][j] !== 0) {
                if (col.has(board[i][j])) return false;
                col.add(board[i][j]);
            }
        }
    }
    
    // 检查3x3子方格
    for (let box = 0; box < 9; box++) {
        const subbox = new Set();
        const rowStart = Math.floor(box / 3) * 3;
        const colStart = (box % 3) * 3;
        
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                const value = board[rowStart + i][colStart + j];
                if (value !== 0) {
                    if (subbox.has(value)) return false;
                    subbox.add(value);
                }
            }
        }
    }
    
    return true;
}

// 回溯算法求解数独
function solve(board) {
    // 找到一个空格 (0表示空格)
    let emptySpot = findEmptySpot(board);
    if (!emptySpot) {
        return true; // 没有空格，说明解决成功
    }
    
    const [row, col] = emptySpot;
    
    // 尝试1-9的每个数字
    for (let num = 1; num <= 9; num++) {
        if (isValid(board, row, col, num)) {
            // 放置数字
            board[row][col] = num;
            
            // 递归继续解决剩余部分
            if (solve(board)) {
                return true;
            }
            
            // 如果无法继续，回溯
            board[row][col] = 0;
        }
    }
    
    return false; // 触发回溯
}

// 在数独中找到一个空格
function findEmptySpot(board) {
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            if (board[i][j] === 0) {
                return [i, j];
            }
        }
    }
    return null; // 没有空格
}

// 检查在特定位置放置数字是否有效
function isValid(board, row, col, num) {
    // 检查同一行
    for (let j = 0; j < 9; j++) {
        if (board[row][j] === num) {
            return false;
        }
    }
    
    // 检查同一列
    for (let i = 0; i < 9; i++) {
        if (board[i][col] === num) {
            return false;
        }
    }
    
    // 检查3x3子方格
    const boxRow = Math.floor(row / 3) * 3;
    const boxCol = Math.floor(col / 3) * 3;
    
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            if (board[boxRow + i][boxCol + j] === num) {
                return false;
            }
        }
    }
    
    return true; // 有效
}

document.getElementById('pasteArea').addEventListener('paste', function (e) {
    // 阻止默认的粘贴行为
    e.preventDefault();

    // 检查粘贴的数据是否包含图片
    if (e.clipboardData && e.clipboardData.items) {
        const items = e.clipboardData.items;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
                resetWorkspace();
                const blob = items[i].getAsFile();
                rawImage.src = URL.createObjectURL(blob);
                imageElm.src = rawImage.src;
                
                const statusElm = document.getElementById('boardStatus');
                statusElm.innerHTML = "处理中...";
                statusElm.style.backgroundColor = "rgba(52, 152, 219, 0.2)"; // 蓝色背景
                statusElm.style.color = "#2980b9";
                break
            }
        }
    }
});

document.getElementById('modelFile').addEventListener('change', function (e) {
    if ((e.target.files) && (e.target.files[0].name.endsWith(".onnx"))) {
        const statusElement = document.getElementById('onnxStatus');
        if (statusElement) {
            statusElement.innerHTML = '正在加载本地OCR模型...';
        }
        
        window.modelPath = URL.createObjectURL(e.target.files[0]);
        // 本地模型文件已选择，触发初始化
        if (window.ort) {
            init();
        }
    }
});

rawImage.onload = async function () {
    const mat = cv.imread(rawImage);
    URL.revokeObjectURL(rawImage.src);

    // 图片预处理，二值化;
    const thresh = new cv.Mat();
    cv.cvtColor(mat, thresh, cv.COLOR_RGBA2GRAY, 0);
    cv.threshold(thresh, thresh, 180, 255, cv.THRESH_BINARY_INV);

    // 找到网格线   
    const lines = findLines(thresh);
    if (!((lines.h.length === 10) && (lines.v.length === 10))) {
        statusElm.innerHTML = '查找网格线出错';
        return null;
    }

    let result = {};
    await recognizeByRow(thresh, lines).then((ret) => { result = ret; })
        .catch((err) => {
            statusElm.innerHTML = "整行识别失败...";
            console.error('整行识别发生错误:', err);
        });

    if (result.full) {
        fillIn(result.board, cells);
        statusElm.innerHTML = "";
        return null;
    }

    boxes = findBoxes(lines);
    await recognizeByBoxes(thresh, boxes).then((ret) => { result = ret; statusElm.innerHTML = ""; })
        .catch((err) => {
            statusElm.innerHTML = "单字识别失败...";
            console.error('单字识别发生错误:', err);
        });;
    thresh.delete();
    mat.delete();

    fillIn(result.board, cells);
};

checkPaddleOCRLoaded();