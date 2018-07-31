$(document).ready(function() {

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