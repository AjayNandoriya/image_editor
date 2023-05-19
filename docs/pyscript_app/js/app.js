function loadImg() {
    let imgElement = document.getElementById("imageSrc");
    let inputElement = document.getElementById("fileInput");
    // imgElement.onload = function () {
    //   let mat = cv.imread(imgElement);
    // };
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

  function base64ToArrayBuffer(base64) {
    var binary_string = window.atob(base64);
    var len = binary_string.length;
    var bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) {
        bytes[i] = binary_string.charCodeAt(i);
    }
    return bytes.buffer;
}

let read_img_btn = document.querySelector('#readImg');
read_img_btn.addEventListener('click',(e)=>{
    let c = document.querySelector('#board');
    let b64 = c.toDataURL();
    let base64 = b64.slice(22);
    let arr = base64ToArrayBuffer(base64);
    console.log(arr);
});

loadImg();