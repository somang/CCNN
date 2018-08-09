$(document).ready(function() {
<<<<<<< HEAD
    var sourcename = 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4';
=======
>>>>>>> 557e6f0eda4642d9416851358f88d1caedd58840

  // query string: ?foo=lorem&bar=&baz
  var speed = getQueryVariable('s');
  var delay = getQueryVariable('d'); 
  var missing_words = getQueryVariable('m'); 
  var paraphrased = getQueryVariable('p');
  console.log(speed, delay, missing_words, paraphrased);
  

});


function getQueryVariable(variable)
{
  var query = window.location.search.substring(1);
  var vars = query.split("&");
  for (var i=0;i<vars.length;i++) {
          var pair = vars[i].split("=");
          if(pair[0] == variable){return pair[1];}
  }
  return(false);
}