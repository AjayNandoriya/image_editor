

function loadImg() {

    let imgElement = document.getElementById("imageSrc")
    let inputElement = document.getElementById("fileInput");
    imgElement.onload = function() {
        let mat = cv.imread(imgElement);
      }
    inputElement.addEventListener("change", (e) => {
        imgElement.src = URL.createObjectURL(e.target.files[0]);
        
    }, false);

    imgElement.onload = function() {
        let mat = cv.imread(imgElement);
        let dsize = new cv.Size(256, 256);
        cv.resize(mat, mat, dsize, 0, 0, cv.INTER_AREA);
        cv.imshow('board', mat);

        mat.delete();
      };
      
}




class DrawingBoard {
    constructor(root_id, canvas_id, type_id, info_id) {
        let H = 256;
        let W = 256;
        this.root = document.querySelector(root_id);
        this.canvas = document.querySelector(canvas_id);
        this.canvas.width = W;
        this.canvas.height = H;
        this.type = document.querySelector(type_id);
        this.info_txt = document.querySelector(info_id);
        this.draw = false;
        console.log(this.type);
        this.contours = [];
        this.canvass = []
        for (let i = 0; i < this.type.length; i++) {
            this.contours.push([]);
            let canvas = document.createElement('CANVAS');
            canvas.id = `mask_${i}`;
            canvas.height = H;
            canvas.width = W;
            // canvas.style.display = 'none';
            this.root.appendChild(canvas);
            this.canvass.push(canvas);

        }

        this.canvas.onmouseover = (e) => {
            let { x, y } = this.getClientOffset(e);
            this.print_x_y(x, y);

        }

        this.canvas.onmousemove = (e) => {
            let { x, y } = this.getClientOffset(e);
            this.print_x_y(x, y);
            if (!this.draw) {
                return;
            }
            let contour = this.contours[this.type.selectedIndex];
            let color = this.type.value;
            let old = contour[contour.length - 1];
            this.drawline(this.canvas, old.x, old.y, x, y, color);
            this.drawline(this.canvass[this.type.selectedIndex], old.x, old.y, x, y, color);
            contour.push({ x, y });
        }

        this.canvas.onmousedown = (e) => {
            this.draw = true;
            let { x, y } = this.getClientOffset(e);
            let contour = this.contours[this.type.selectedIndex];
            contour.push({ x, y });
        }

        this.canvas.onmouseup = (e) => {
            this.draw = false;
            // let {x,y} = this.getClientOffset(e);
            // let contour = this.contours[this.type.selectedIndex];
            // contour.push({x,y});
        }

    }

    print_x_y(x, y) {
        this.info_txt.value = `x:${x.toFixed(2)} , y:${y.toFixed(2)}`;
    }

    getClientOffset = (event) => {
        const { pageX, pageY } = event.touches ? event.touches[0] : event;
        const x = pageX - this.canvas.offsetLeft;
        const y = pageY - this.canvas.offsetTop;

        return {
            x,
            y
        }
    }
    drawline(canvas, x1, y1, x2, y2, color) {
        let ctx = canvas.getContext('2d');
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = '1';
        ctx.lineCap = "rounded";
        ctx.moveTo(x1, y1);
        // Specify where the line ends
        ctx.lineTo(x2, y2);
        // Draw the line
        ctx.stroke();
    }

    segment_image(main_canvas, fg_canvas, bg_canvas, out_canvas){
        for (let i = 0; i < src.rows; i++) {
            for (let j = 0; j < src.cols; j++) {
                if (mask.ucharPtr(i, j)[0] == 0 || mask.ucharPtr(i, j)[0] == 2) {
                    src.ucharPtr(i, j)[0] = 0;
                    src.ucharPtr(i, j)[1] = 0;
                    src.ucharPtr(i, j)[2] = 0;
                }
            }
        }
        cv.grabCut(main_canvas, mask, rect, bgdModel, fgdModel, 1, cv.GC_INIT_WITH_MASK);
    }

}

let drawingBoard = new DrawingBoard('#drawing_board', '#board', '#contourcolor', '#board_info');
loadImg();