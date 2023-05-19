
var canvas = document.getElementById('ref');
var context = canvas.getContext("2d");
var fileUpload = document.getElementById('fileUpload');

var ref_img;
var ref_img_scale = 1.0;
var ref_img_scale_range = [0.1, 10.0];
var ref_img_scale = 1.0;
var ref_img_offset = [0,0];
var ref_img_mat;
var test_img_mat;
function readImage() {
    if (this.files && this.files[0]) {
        var FR = new FileReader();
        FR.onload = function (e) {
            var img = new Image();
            img.src = e.target.result;
            img.onload = function () {
                ref_img = img;
                ref_img_mat = cv.imread(ref_img);
                test_img_mat = new cv.Mat();
                let kernel = cv.Mat.eye(3, 3, cv.CV_32FC1);
                let anchor = new cv.Point(-1, -1);
                // You can try more different parameters
                cv.filter2D(ref_img_mat, test_img_mat, cv.CV_8U, kernel, anchor, 0, cv.BORDER_DEFAULT);


                let M = [[1,0,0],[0,1,0]];
                draw_on_canvas(canvas, ref_img_mat, M);

                let test_canvas = document.getElementById('test');
                draw_on_canvas(test_canvas, test_img_mat, M);
            };
        };
        FR.readAsDataURL(this.files[0]);
    }
}

fileUpload.onchange = readImage;

function imagedata_to_image(imagedata) {
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    canvas.width = imagedata.width;
    canvas.height = imagedata.height;
    ctx.putImageData(imagedata, 0, 0);

    var image = new Image();
    image.src = canvas.toDataURL();
    return image;
}

function draw_on_canvas(canvas, img, M){
    console.log('M',M);
    let context = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    context.clearRect(0,0,512,512);

    cv.cvtColor(img,img,cv.COLOR_RGB2RGBA);
    let imgData=new ImageData(new Uint8ClampedArray(img.data),img.cols,img.rows);
    let w = (M[0][0])*imgData.width;
    let h = (M[1][1])*imgData.height;
    let x1 = 0.5*(1 - M[0][0])*imgData.width + (1-M[0][0])*(M[0][2] - 0.5)*imgData.width/2 ;
    let y1 = 0.5*(1 - M[1][1])*imgData.height + (1-M[1][1])*(M[1][2] - 0.5)*imgData.height/2;
    
    canvas.height = h;
    canvas.width = w;
    
    img2 = imagedata_to_image(imgData);
    // cv.imshow(canvas, img);
    context.drawImage(img2, x1, y1, w, h);
    // context.putImageData(img, x1, y1, w, h);
}

function adjustZoom(e){
    const SENSTIVITY = 0.001;
    

    // console.log('mouse', e.clientX, e.clientY);
    // console.log('relative', e.clientX - rect.left,e.clientY - rect.top);
    // console.log(canvas.width, canvas.height);

    // context.save();
    ref_img_scale += e.deltaY*SENSTIVITY;
    if ( ref_img_scale < ref_img_scale_range[0]){
        ref_img_scale = ref_img_scale_range[0];
    }
    else if(ref_img_scale > ref_img_scale_range[1]){
        ref_img_scale = ref_img_scale_range[1];
    }
    
    const rect = e.target.getBoundingClientRect();
    let offsetX = (e.clientX - rect.left)/rect.width ;
    let offsetY = (e.clientY - rect.top)/rect.height ;

    // console.log('offset',offsetX, offsetY);
    let M = [[ref_img_scale,0,offsetX],[0,ref_img_scale,offsetY]];
    let ref_canvas = document.getElementById('ref');
    draw_on_canvas(ref_canvas, ref_img_mat, M);

    let test_canvas = document.getElementById('test');
    draw_on_canvas(test_canvas, test_img_mat, M);
    
    
}

canvas.onwheel = (e) => adjustZoom(e);
let test_canvas = document.getElementById('test');
test_canvas.onwheel = (e) => adjustZoom(e);



function sync_scroll(ids){
    for( let i1=0;i1<ids.length;i1++){
        for( let i2=0;i2<ids.length;i2++){
            if(i1 == i2){
                continue;
            }
            $('#' + ids[i1]).on('scroll', function () {
                $('#' + ids[i2]).scrollTop($(this).scrollTop());
                $('#' + ids[i2]).scrollLeft($(this).scrollLeft());
            });
            
        }
    }
}

sync_scroll(['refdiv', 'testdiv']);