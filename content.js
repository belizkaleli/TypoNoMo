// MIT License
// Andrej Karpathy

var svmjs = (function(exports){

  /*
    This is a binary SVM and is trained using the SMO algorithm.
    Reference: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)

    Simple usage example:
    svm = svmjs.SVM();
    svm.train(data, labels);
    testlabels = svm.predict(testdata);
  */
  var SVM = function(options) {
  }

  SVM.prototype = {

    // data is NxD array of floats. labels are 1 or -1.
    train: function(data, labels, options) {

      // we need these in helper functions
      this.data = data;
      this.labels = labels;

      // parameters
      options = options || {};
      var C = options.C || 1.0; // C value. Decrease for more regularization
      var tol = options.tol || 1e-4; // numerical tolerance. Don't touch unless you're pro
      var alphatol = options.alphatol || 1e-7; // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
      var maxiter = options.maxiter || 10000; // max number of iterations
      var numpasses = options.numpasses || 10; // how many passes over data with no change before we halt? Increase for more precision.

      // instantiate kernel according to options. kernel can be given as string or as a custom function
      var kernel = linearKernel;
      this.kernelType = "linear";
      if("kernel" in options) {
        if(typeof options.kernel === "string") {
          // kernel was specified as a string. Handle these special cases appropriately
          if(options.kernel === "linear") {
            this.kernelType = "linear";
            kernel = linearKernel;
          }
          if(options.kernel === "rbf") {
            var rbfSigma = options.rbfsigma || 0.5;
            this.rbfSigma = rbfSigma; // back this up
            this.kernelType = "rbf";
            kernel = makeRbfKernel(rbfSigma);
          }
        } else {
          // assume kernel was specified as a function. Let's just use it
          this.kernelType = "custom";
          kernel = options.kernel;
        }
      }

      // initializations
      this.kernel = kernel;
      this.N = data.length; var N = this.N;
      this.D = data[0].length; var D = this.D;
      this.alpha = zeros(N);
      this.b = 0.0;
      this.usew_ = false; // internal efficiency flag

      // Cache kernel computations to avoid expensive recomputation.
      // This could use too much memory if N is large.
      if (options.memoize) {
        this.kernelResults = new Array(N);
        for (var i=0;i<N;i++) {
          this.kernelResults[i] = new Array(N);
          for (var j=0;j<N;j++) {
            this.kernelResults[i][j] = kernel(data[i],data[j]);
          }
        }
      }

      // run SMO algorithm
      var iter = 0;
      var passes = 0;
      while(passes < numpasses && iter < maxiter) {

        var alphaChanged = 0;
        for(var i=0;i<N;i++) {

          var Ei= this.marginOne(data[i]) - labels[i];
          if( (labels[i]*Ei < -tol && this.alpha[i] < C)
           || (labels[i]*Ei > tol && this.alpha[i] > 0) ){

            // alpha_i needs updating! Pick a j to update it with
            var j = i;
            while(j === i) j= randi(0, this.N);
            var Ej= this.marginOne(data[j]) - labels[j];

            // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
            ai= this.alpha[i];
            aj= this.alpha[j];
            var L = 0; var H = C;
            if(labels[i] === labels[j]) {
              L = Math.max(0, ai+aj-C);
              H = Math.min(C, ai+aj);
            } else {
              L = Math.max(0, aj-ai);
              H = Math.min(C, C+aj-ai);
            }

            if(Math.abs(L - H) < 1e-4) continue;

            var eta = 2*this.kernelResult(i,j) - this.kernelResult(i,i) - this.kernelResult(j,j);
            if(eta >= 0) continue;

            // compute new alpha_j and clip it inside [0 C]x[0 C] box
            // then compute alpha_i based on it.
            var newaj = aj - labels[j]*(Ei-Ej) / eta;
            if(newaj>H) newaj = H;
            if(newaj<L) newaj = L;
            if(Math.abs(aj - newaj) < 1e-4) continue;
            this.alpha[j] = newaj;
            var newai = ai + labels[i]*labels[j]*(aj - newaj);
            this.alpha[i] = newai;

            // update the bias term
            var b1 = this.b - Ei - labels[i]*(newai-ai)*this.kernelResult(i,i)
                     - labels[j]*(newaj-aj)*this.kernelResult(i,j);
            var b2 = this.b - Ej - labels[i]*(newai-ai)*this.kernelResult(i,j)
                     - labels[j]*(newaj-aj)*this.kernelResult(j,j);
            this.b = 0.5*(b1+b2);
            if(newai > 0 && newai < C) this.b= b1;
            if(newaj > 0 && newaj < C) this.b= b2;

            alphaChanged++;

          } // end alpha_i needed updating
        } // end for i=1..N

        iter++;
        //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
        if(alphaChanged == 0) passes++;
        else passes= 0;

      } // end outer loop

      // if the user was using a linear kernel, lets also compute and store the
      // weights. This will speed up evaluations during testing time
      if(this.kernelType === "linear") {

        // compute weights and store them
        this.w = new Array(this.D);
        for(var j=0;j<this.D;j++) {
          var s= 0.0;
          for(var i=0;i<this.N;i++) {
            s+= this.alpha[i] * labels[i] * data[i][j];
          }
          this.w[j] = s;
          this.usew_ = true;
        }
      } else {

        // okay, we need to retain all the support vectors in the training data,
        // we can't just get away with computing the weights and throwing it out

        // But! We only need to store the support vectors for evaluation of testing
        // instances. So filter here based on this.alpha[i]. The training data
        // for which this.alpha[i] = 0 is irrelevant for future.
        var newdata = [];
        var newlabels = [];
        var newalpha = [];
        for(var i=0;i<this.N;i++) {
          //console.log("alpha=%f", this.alpha[i]);
          if(this.alpha[i] > alphatol) {
            newdata.push(this.data[i]);
            newlabels.push(this.labels[i]);
            newalpha.push(this.alpha[i]);
          }
        }

        // store data and labels
        this.data = newdata;
        this.labels = newlabels;
        this.alpha = newalpha;
        this.N = this.data.length;
        //console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
      }

      var trainstats = {};
      trainstats.iters= iter;
      return trainstats;
    },

    // inst is an array of length D. Returns margin of given example
    // this is the core prediction function. All others are for convenience mostly
    // and end up calling this one somehow.
    marginOne: function(inst) {

      var f = this.b;
      // if the linear kernel was used and w was computed and stored,
      // (i.e. the svm has fully finished training)
      // the internal class variable usew_ will be set to true.
      if(this.usew_) {

        // we can speed this up a lot by using the computed weights
        // we computed these during train(). This is significantly faster
        // than the version below
        for(var j=0;j<this.D;j++) {
          f += inst[j] * this.w[j];
        }

      } else {

        for(var i=0;i<this.N;i++) {
          f += this.alpha[i] * this.labels[i] * this.kernel(inst, this.data[i]);
        }
      }

      return f;
    },

    predictOne: function(inst) {
      return this.marginOne(inst) > 0 ? 1 : -1;
    },

    // data is an NxD array. Returns array of margins.
    margins: function(data) {

      // go over support vectors and accumulate the prediction.
      var N = data.length;
      var margins = new Array(N);
      for(var i=0;i<N;i++) {
        margins[i] = this.marginOne(data[i]);
      }
      return margins;

    },

    kernelResult: function(i, j) {
      if (this.kernelResults) {
        return this.kernelResults[i][j];
      }
      return this.kernel(this.data[i], this.data[j]);
    },

    // data is NxD array. Returns array of 1 or -1, predictions
    predict: function(data) {
      var margs = this.margins(data);
      for(var i=0;i<margs.length;i++) {
        margs[i] = margs[i] > 0 ? 1 : -1;
      }
      return margs;
    },

    // THIS FUNCTION IS NOW DEPRECATED. WORKS FINE BUT NO NEED TO USE ANYMORE.
    // LEAVING IT HERE JUST FOR BACKWARDS COMPATIBILITY FOR A WHILE.
    // if we trained a linear svm, it is possible to calculate just the weights and the offset
    // prediction is then yhat = sign(X * w + b)
    getWeights: function() {

      // DEPRECATED
      var w= new Array(this.D);
      for(var j=0;j<this.D;j++) {
        var s= 0.0;
        for(var i=0;i<this.N;i++) {
          s+= this.alpha[i] * this.labels[i] * this.data[i][j];
        }
        w[j]= s;
      }
      return {w: w, b: this.b};
    },

    toJSON: function() {

      if(this.kernelType === "custom") {
        console.log("Can't save this SVM because it's using custom, unsupported kernel...");
        return {};
      }

      json = {}
      json.N = this.N;
      json.D = this.D;
      json.b = this.b;

      json.kernelType = this.kernelType;
      if(this.kernelType === "linear") {
        // just back up the weights
        json.w = this.w;
      }
      if(this.kernelType === "rbf") {
        // we need to store the support vectors and the sigma
        json.rbfSigma = this.rbfSigma;
        json.data = this.data;
        json.labels = this.labels;
        json.alpha = this.alpha;
      }

      return json;
    },

    fromJSON: function(json) {

      this.N = json.N;
      this.D = json.D;
      this.b = json.b;

      this.kernelType = json.kernelType;
      if(this.kernelType === "linear") {

        // load the weights!
        this.w = json.w;
        this.usew_ = true;
        this.kernel = linearKernel; // this shouldn't be necessary
      }
      else if(this.kernelType == "rbf") {

        // initialize the kernel
        this.rbfSigma = json.rbfSigma;
        this.kernel = makeRbfKernel(this.rbfSigma);

        // load the support vectors
        this.data = json.data;
        this.labels = json.labels;
        this.alpha = json.alpha;
      } else {
        console.log("ERROR! unrecognized kernel type." + this.kernelType);
      }
    }
  }

  // Kernels
  function makeRbfKernel(sigma) {
    return function(v1, v2) {
      var s=0;
      for(var q=0;q<v1.length;q++) { s += (v1[q] - v2[q])*(v1[q] - v2[q]); }
      return Math.exp(-s/(2.0*sigma*sigma));
    }
  }

  function linearKernel(v1, v2) {
    var s=0;
    for(var q=0;q<v1.length;q++) { s += v1[q] * v2[q]; }
    return s;
  }

  // Misc utility functions
  // generate random floating point number between a and b
  function randf(a, b) {
    return Math.random()*(b-a)+a;
  }

  // generate random integer between a and b (b excluded)
  function randi(a, b) {
     return Math.floor(Math.random()*(b-a)+a);
  }

  // create vector of zeros of length n
  function zeros(n) {
    var arr= new Array(n);
    for(var i=0;i<n;i++) { arr[i]= 0; }
    return arr;
  }

  // export public members
  exports = exports || {};
  exports.SVM = SVM;
  exports.makeRbfKernel = makeRbfKernel;
  exports.linearKernel = linearKernel;
  return exports;

})(typeof module != 'undefined' && module.exports);  // add exports to module.exports if in node.js



// DNS-JS.com - Make DNS queries from Javascript
// Copyright 2019 Infinite Loop Development Ltd - InfiniteLoop.ie
// Do not remove this notice.

DNS = {
    QueryType : {
        A : 1,
        NS : 2,
        MD : 3,
        MF : 4,
        CNAME : 5,
        SOA : 6,
        MB : 7,
        MG : 8,
        MR : 9,
        NULL : 10,
        WKS : 11,
        PTR : 12,
        HINFO : 13,
        MINFO : 14,
        MX : 15,
        TXT : 16,
        RP : 17,
        AFSDB : 18,
        AAAA : 28,
        SRV : 33,
        SSHFP : 44,
        RRSIG : 46,
        AXFR : 252,
        ANY : 255,
        URI : 256,
        CAA : 257
    },
    Query: function (domain, type, callback) {
        DNS._callApi({
            Action: "Query",
            Domain: domain,
            Type: type
        },
        callback);
    },
    _callApi: function (request, callback) {
        var xhr = new XMLHttpRequest();
        URL = "https://www.dns-js.com/api.aspx";
        xhr.open("POST", URL, false);
        xhr.onreadystatechange = function () {
            if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
                callback(JSON.parse(xhr.response));
            }
        }
        xhr.send(JSON.stringify(request));
    }
};

class URLElement {
  constructor(url, beginning, middle, end, repetition, preposition, ns, net, co, gov, it, my, no, so, you, to, zip, string) {
		this.url = url;
    this.beginning = beginning;
		this.middle = middle;
		this.end = end;
		this.repetition = repetition;
		this.preposition = preposition;
		this.ns = ns;
		this.net = net;
		this.co = co;
		this.gov = gov;
		this.it = it;
		this.my = my;
		this.no = no;
		this.so = so;
		this.you = you;
		this.to = to;
		this.zip = zip;
		this.string = string;
  }
}

class TextofURL {
	constructor(url, text) {
		this.url = url;
		this.text = text;
	}
}


//CREATE WORD DICTIONARY
var url = chrome.runtime.getURL('./words.txt');
var dictionary_words = new Array();

fetch(url)
    .then((response) => response.text())
    .then((text) => createWordDict(text));

function createWordDict(text) {
	dictionary_words = text.split(/\n|\r/g);
	console.log(dictionary_words);
}

//CREATE TLD DICTIONARY
url = chrome.runtime.getURL('./effective_tld_names.json');
var dictionary_tlds = new Array();

fetch(url)
    .then((response) => response.json())
    .then((json) => createTLDDict(json));

function createTLDDict(json) {
	dictionary_tlds = Object.keys(json);
	console.log(dictionary_tlds);
}

var mymodal;
var spn;
var btnCont;
var btnCancel;
var p;

function createModal() {
  mymodal = document.createElement("div");
  mymodal.setAttribute("id", "myModal");
  mymodal.setAttribute("class", "modal");
  var element = document.getElementById("react-root");
  element.appendChild(mymodal);

  var cntnt = document.createElement("div");
  cntnt.setAttribute("class", "modal-content");
  mymodal.appendChild(cntnt);

  spn = document.createElement("span");
  spn.setAttribute("class", "close");
  var close = document.createTextNode("X");
  spn.appendChild(close);
  cntnt.appendChild(spn);

  p = document.createElement("p");
  var modaltext = document.createTextNode("Placeholder");
  p.appendChild(modaltext);
  cntnt.appendChild(p);

  btnCont = document.createElement("button");
  btnCont.setAttribute("id", "button-continue");
  btnCont.setAttribute("class", "btncontinue");
  var btnContText = document.createTextNode("Continue");
  btnCont.appendChild(btnContText);
  cntnt.appendChild(btnCont);
  btnCancel = document.createElement("button");
  btnCancel.setAttribute("id", "button-cancel");
  btnCancel.setAttribute("class", "btncancel");
  var btnCancelText = document.createTextNode("Cancel");
  cntnt.appendChild(btnCancel);
  btnCancel.appendChild(btnCancelText);
}


var newTweet = true;
var button1;
var ignore = false;
document.addEventListener('keyup', beginTypoCheck, true);
createModal();

function beginTypoCheck(e) {
	console.log('keyup', e.code);

	if (newTweet) {
		button1 = document.getElementsByClassName('css-18t94o4 css-1dbjc4n r-urgr8i r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1w2pmg r-1n0xq6e r-1vuscfd r-1dhvaqw r-1fneopy r-o7ynqc r-6416eg r-lrvibr')[0];

		//css-1dbjc4n r-urgr8i r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1w2pmg r-1n0xq6e r-1vuscfd r-1dhvaqw r-icoktb r-1fneopy r-o7ynqc r-6416eg r-lrvibr
		//css-18t94o4 css-1dbjc4n r-urgr8i r-42olwf r-sdzlij r-1phboty r-rs99b7 r-1w2pmg r-1n0xq6e r-1vuscfd r-1dhvaqw r-1fneopy r-o7ynqc r-6416eg r-lrvibr

		button1.addEventListener('click', TweetButtonClicked, true);
		newTweet = false;
	}
}

function Prefiltering(url) {
	url = url.toLowerCase();
	var otherwords = new Array();
	otherwords = url.match(/(?<=\.)\w+/g);
	var firstword = url.match(/^\w+/)[0];
	var tld = '';
	var tldIsWord;
	var firstwordIsWord;
	var otherwordsAreWords;
	var countOtherword = 0;

	if (url.includes('/') || url.includes('?') || url.includes('www')) {
		return [false, tld, firstword, otherwords];
	}

	if (otherwords.length > 1) {
		//url contains subdomain and/or TLD with a dot in it
		var endingOfUrl = otherwords[otherwords.length - 2] + '.' + otherwords[otherwords.length - 1];

		dictionary_tlds.forEach((item, i) => {
		  if (endingOfUrl.includes(item) && item.length > tld.length && item.includes(otherwords[otherwords.length - 1])) {
				tld = item;
			}
		});
	} else {
		//url is in the form str.str
		tld = otherwords[0];
	}

	if (tld == 'com' || tld == 'org' || tld.includes('.')) {
		return [false, tld, firstword, otherwords];
	}


	dictionary_words.forEach((item, i) => {
		item = item.toLowerCase();
		if (tld == item) {
			//console.log('Tld is a word: ', item);
			tldIsWord = true;
		}
		if (firstword == item || isNaN(firstword) == false) {
			//console.log('firstword is a word: ', item);
			firstwordIsWord = true;
		}
		for (var k = 0; k < otherwords.length - 1; k++) {
			if(otherwords[k] == item || isNaN(otherwords[k]) == false) {
				//console.log(otherwords[k], 'is a word');
				countOtherword++;
			}
		}
	});

	if (tldIsWord) {
		return [true, tld, firstword, otherwords];
	} else {
		return [false, tld, firstword, otherwords];
	}



}

function calculateFeatures(firstword, otherwords, i, urls, tld, url_elements_arr, textofurl_elements_arr) {
	var url_element = new URLElement();
	url_element.url = urls[i];
	var s = urls[i];
	text = textofurl_elements_arr[i].text;
	var url_length = urls[i].length;
	var regex = new RegExp(s);
	var startof_url = regex.exec(text).index;
	var text_length = text.length;
	//Beginning, End, Middle
	if (startof_url == text_length - url_length) {
		url_element.end = 1;
		url_element.beginning = 0;
		url_element.middle = 0;
	} else if (startof_url == 0) {
		url_element.end = 0;
		url_element.beginning = 1;
		url_element.middle = 0;
	} else {
		url_element.end = 0;
		url_element.beginning = 0;
		url_element.middle = 1;
	}

	//Repetition
	regex = new RegExp(s, 'g');
	var matches = [...text.matchAll(regex)];
	if (matches.length > 1) {
		url_element.repetition = 1;
	} else {
		url_element.repetition = 0;
	}

	//Preposition
	regex = new RegExp('([^ \r\n]+) ' + urls[i], 'g');
	matches = [...text.matchAll(regex)];
	if (matches.length == 0) {
		url_element.preposition = 1;
	} else {
		for (match of matches) {
			//console.log('match: ', match);
			var previous_word = match[1].toLowerCase();
			//console.log('previous_word: ', previous_word);
			if (previous_word == 'on'|| previous_word == 'via' || previous_word == 'to' || previous_word == 'at' || (previous_word.includes('website')) || (previous_word.includes(':')) || (previous_word.includes('visit'))) {
				url_element.preposition = 0;
				break;
			} else {
				url_element.preposition = 1;
			}
		}
	}

	//tlds
	if (tld == 'net') {
		url_element.net = 1;
	} else {
		url_element.net = 0;
	}

  if (tld == 'co') {
		url_element.co = 1;
	} else {
		url_element.co = 0;
	}

	if (tld == 'gov') {
		url_element.gov = 1;
	} else {
		url_element.gov = 0;
	}

	if (tld == 'it') {
		url_element.it = 1;
	} else {
		url_element.it = 0;
	}

	if (tld == 'my') {
		url_element.my = 1;
	} else {
		url_element.my = 0;
	}

	if (tld == 'no') {
		url_element.no = 1;
	} else {
		url_element.no = 0;
	}

	if (tld == 'so') {
		url_element.so = 1;
	} else {
		url_element.so = 0;
	}

	if (tld == 'you') {
		url_element.you = 1;
	} else {
		url_element.you = 0;
	}

	if (tld == 'to') {
		url_element.to = 1;
	} else {
		url_element.to = 0;
	}

	if (tld == 'zip') {
		url_element.zip = 1;
	} else {
		url_element.zip = 0;
	}

	//STRING
	var count = 0;
	if (firstword.length < 2) {
		otherwords.forEach((item, i) => {
			if (item.length < 2) {
				count++;
			}
		});
	}

	var countOtherword = 0;
	var firstwordIsWord = false;
	var otherwordsAreWords = false;
	var firstwordisNumber = true;
	var otherwordsAreNumbers = true;
	var firstwordCamelCase = false;
	var otherwordsCamelCase = false;
	var urlDotCapital = false;
	dictionary_words.forEach((item, i) => {
		item = item.toLowerCase();
		if (firstword == item) {
			firstwordIsWord = true;
		}
		for (var k = 0; k < otherwords.length - 1; k++) {
			if(otherwords[k] == item) {
				countOtherword++;
			}
		}
	});
	if (countOtherword == otherwords.length - 1) {
		otherwordsAreWords = true;
	}

	if (isNaN(firstword)) {
		firstwordisNumber = false;
	}
	otherwords.forEach((item, i) => {
		if (isNaN(item)) {
			otherwordsAreNumbers = false;
		}
	});

	if (firstword.match(/[a-z][A-Z]/g) != null) {
		firstwordCamelCase = true;
	}
	otherwords.forEach((item, i) => {
		if (item.match(/[a-z][A-Z]/g) != null) {
			otherwordsCamelCase = true;
		}
	});
	if (urls[i].match(/[a-z]\.[A-Z]/g) != null) {
		var urlDotCapital = true;
	}

	//conditions
	url_element.string = 0;
	if (count == otherwords.length - 1) {
		url_element.string = 1;
	} else if (containsDash()) {
		url_element.string = 0;
	} else if ((firstwordIsWord == false && firstwordisNumber == false) || (otherwordsAreWords == false  && otherwordsAreWords == false)) {
		url_element.string = 0;
	} else if (firstwordisNumber && otherwords.length == 1) {
		url_element.string = 1;
	} else if (firstwordisNumber && otherwordsAreNumbers) {
		url_element.string = 1;
	} else if (firstwordisNumber && otherwordsAreWords && otherwordsCamelCase == false) {
		url_element.string = 1;
	} else if (firstwordIsWord && firstwordCamelCase == false && otherwords.length == 1 && urlDotCapital) {
		url_element.string = 1;
	} else if (otherwordsAreWords && otherwordsCamelCase == false && firstwordisNumber) {
		url_element.string = 1;
	} else if (otherwordsAreWords && otherwordsCamelCase == false && firstwordIsWord && firstwordCamelCase == false) {
		url_element.string = 1;
	}


  function containsDash() {
		if (firstword.includes('-')) {
			return true;
		} else {
			otherwords.forEach((item, i) => {
				if (item.includes('-')) {
					return true;
				}
			});
		}
	}

	//NS
	DNS.Query(urls[i], DNS.QueryType.NS, function(data) {
		//console.log('ns: ', data);
		if (data.length == 0) {
			url_element.ns = 1;
		} else {
			url_element.ns = 0;
		}
		url_elements_arr.push(url_element);
	});
}



function TweetButtonClicked(event) {
  var start = Date.now();
  console.log('Tweet button clicked:', start);
	if (ignore == false) {
		event.stopPropagation();
		//console.log("Button1 innerText: " + button1.innerText);
		newTweet = true;
		//alert("Button1 clicked");
		var textElements = document.getElementsByClassName("public-DraftStyleDefault-block public-DraftStyleDefault-ltr");
		var textofurl_elements_arr = new Array();
		var urls = new Array();
		for (textElement of textElements) {
			var text = textElement.innerText;
			var lenChildNodes = textElement.children.length;
			for (var i = 0; i < lenChildNodes; i++) {
				if (textElement.children[i].style.cssText != "") {
					var textofurl_element = new TextofURL();
					var u = textElement.children[i].innerText;
					u = u.replace('https://', '');
					u = u.replace('http://', '');
					textofurl_element.url = u;
					textofurl_element.text = text;
					if (!urls.includes(u)) {
						urls.push(u);
						textofurl_elements_arr.push(textofurl_element);
					}
				}
			}
		}
		console.log('urls: ', urls);
		var url_elements_arr = new Array();

		for (i = 0; i < urls.length; i++) {

			var results = Prefiltering(urls[i]);
			var isPossibleTypo = results[0];
			var tld = results[1];
			var firstword = results[2];
			var otherwords = results[3];
			console.log('isPossibleTypo:', isPossibleTypo, 'tld:', tld);
			if (isPossibleTypo) {
				calculateFeatures(firstword, otherwords, i, urls, tld, url_elements_arr, textofurl_elements_arr);
			}
		}
		console.log('URL Elements: ', url_elements_arr);

		predict(url_elements_arr, event);

	}


	function predict(url_elements_arr, event) {
		var formatted_url_elements_arr = new Array();
		url_elements_arr.forEach((item, i) => {
			var url_element = new Array;
			url_element.push(item.ns);
			url_element.push(item.preposition);
			url_element.push(item.string);
			url_element.push(item.repetition);
			url_element.push(item.beginning);
			url_element.push(item.end);
			url_element.push(item.middle);
			url_element.push(item.net);
			url_element.push(item.co);
			url_element.push(item.gov);
			url_element.push(item.it);
			url_element.push(item.my);
			url_element.push(item.no);
			url_element.push(item.so);
			url_element.push(item.you);
			url_element.push(item.to);
			url_element.push(item.zip);
			formatted_url_elements_arr.push(url_element);
		});

		//console.log(formatted_url_elements_arr);
		url = chrome.runtime.getURL('./created_model.json');

		fetch(url)
		    .then((response) => response.json())
		    .then((json) => predictTypos(json, event));

		function predictTypos(json, event) {
			var mysvm = new svmjs.SVM();
		  mysvm.fromJSON(json);
		  //console.log(mysvm);
			var predictions = mysvm.predict(formatted_url_elements_arr);
			console.log('my prediction (1:Typo, -1:Not Typo): ', predictions);
			var typo_arr = new Array();
			var alertText = '';
			predictions.forEach((item, i) => {
				if (item == 1) {
					typo_arr.push(url_elements_arr[i]);
				}
			});
			if (typo_arr.length != 0) {
				typo_arr.forEach((item, i) => {
					alertText += item.url + ' ';
				});

        var stop = Date.now();
        console.log('Dialog created:', stop);
        p.innerHTML = 'WARNING! <br><br> You are about to post following link(s): ' + alertText;

        console.log('Processing Time:', (stop - start) / 1000, ' seconds');

        mymodal.style.display = "block";
        spn.onclick = function() {
          mymodal.style.display = "none";
        }

        btnCancel.onclick = function() {
          mymodal.style.display = "none";
        }

        btnCont.onclick = function() {
          mymodal.style.display = "none";
          ignore = true;
          button1.dispatchEvent(event);
          ignore = false;
        }
			} else {
        mymodal.style.display = "none";
        ignore = true;
        button1.dispatchEvent(event);
        var stop = Date.now();
        console.log('Tweet posted:', stop);
        console.log('Processing Time:', (stop - start) / 1000, ' seconds');
        ignore = false;
      }
		}
	}

}
