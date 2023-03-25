function loadImg() {
  let imgElement = document.getElementById("imageSrc");
  let inputElement = document.getElementById("fileInput");
  imgElement.onload = function () {
    let mat = cv.imread(imgElement);
  };
  inputElement.addEventListener(
    "change",
    (e) => {
      imgElement.src = URL.createObjectURL(e.target.files[0]);
    },
    false
  );

  imgElement.onload = function () {
    let mat = cv.imread(imgElement);
    let dsize = new cv.Size(256, 256);
    cv.resize(mat, mat, dsize, 0, 0, cv.INTER_AREA);
    cv.imshow("board", mat);

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
    this.canvass = [];
    this.mask_canvas = document.createElement("CANVAS");
    this.mask_canvas.id = "mask";
    this.mask_canvas.height = H;
    this.mask_canvas.width = W;
    let ctx = this.mask_canvas.getContext("2d");

    // Now draw!
    ctx.fillStyle = "rgb(2,2,2)";
    ctx.fillRect(0, 0, this.mask_canvas.width, this.mask_canvas.height);
    for (let i = 0; i < this.type.length; i++) {
      this.contours.push([]);
      let canvas = document.createElement("CANVAS");
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
    };

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
      this.drawline(
        this.canvass[this.type.selectedIndex],
        old.x,
        old.y,
        x,
        y,
        color
      );
      if (this.type.selectedIndex == 0) {
        this.drawline(this.mask_canvas, old.x, old.y, x, y, "rgb(0,0,0)");
      } else if (this.type.selectedIndex == 1) {
        this.drawline(this.mask_canvas, old.x, old.y, x, y, "rgb(1,1,1)");
      }
      contour.push({ x, y });
    };

    this.canvas.onmousedown = (e) => {
      this.draw = true;
      let { x, y } = this.getClientOffset(e);
      let contour = this.contours[this.type.selectedIndex];
      contour.push({ x, y });
    };

    this.canvas.onmouseup = (e) => {
      this.draw = false;
      // let {x,y} = this.getClientOffset(e);
      // let contour = this.contours[this.type.selectedIndex];
      // contour.push({x,y});
    };
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
      y,
    };
  };
  drawline(canvas, x1, y1, x2, y2, color) {
    let ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = "1";
    ctx.lineCap = "rounded";
    ctx.moveTo(x1, y1);
    // Specify where the line ends
    ctx.lineTo(x2, y2);
    // Draw the line
    ctx.stroke();
  }

  segment_image() {
    let main_img = "imageSrc";
    let fg_canvas_id = "mask_0";
    let bg_canvas_id = "mask_1";
    let out_canvas_id = "boardOut";
    let main_img_mat = cv.imread(main_img);
    let dsize = new cv.Size(256, 256);
    cv.resize(main_img_mat, main_img_mat, dsize, 0, 0, cv.INTER_AREA);
    cv.cvtColor(main_img_mat, main_img_mat, cv.COLOR_RGBA2RGB);
    let mask = cv.imread(this.mask_canvas);
    mask.convertTo(mask, cv.CV_8U, 100, 0);
    cv.cvtColor(mask, mask, cv.COLOR_RGBA2GRAY);

    let fg = cv.imread(document.querySelector("#mask_0"));
    let bg = cv.imread(document.querySelector("#mask_1"));
    cv.cvtColor(fg, fg, cv.COLOR_RGBA2GRAY);
    cv.cvtColor(bg, bg, cv.COLOR_RGBA2GRAY);
    cv.threshold(fg, fg, 1, 1, cv.THRESH_BINARY);
    cv.threshold(bg, bg, 1, 1, cv.THRESH_BINARY);

    cv.addWeighted(fg, 1.0, bg, 2, 1, mask);
    for (let r = 0; r < mask.rows; r++) {
      for (let c = 0; c < mask.cols; c++) {
        let pixel = mask.ucharPtr(r, c);
        if (fg.ucharPtr(r, c)[0] == 1) {
          pixel[0] = 1;
        } else if (bg.ucharPtr(r, c)[0] == 1) {
          pixel[0] = 0;
        } else {
          pixel[0] = 2;
        }
      }
    }
    mask.convertTo(mask, cv.CV_8U, 1, 0);
    let bgdModel = new cv.Mat();
    let fgdModel = new cv.Mat();
    let rect = new cv.Rect(0, 0, 100, 100);
    cv.grabCut(
      main_img_mat,
      mask,
      rect,
      bgdModel,
      fgdModel,
      10,
      cv.GC_INIT_WITH_MASK
    );

    // mask.convertTo(mask, cv.CV_8U, -1, 2);
    mask.convertTo(mask, cv.CV_8U, 100, 0);
    cv.cvtColor(main_img_mat, main_img_mat, cv.COLOR_RGB2GRAY);
    cv.addWeighted(main_img_mat, 0.5, mask, 0.5, 1, mask);
    cv.imshow(out_canvas_id, mask);
    mask.delete();
    main_img_mat.delete();
    fg.delete();
    bg.delete();
  }
}

let drawingBoard = new DrawingBoard(
  "#drawing_board",
  "#board",
  "#contourcolor",
  "#board_info"
);
loadImg();

$("#segment").click(() => {
  drawingBoard.segment_image();
});
