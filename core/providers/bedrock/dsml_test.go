package bedrock

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/bytedance/sonic"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

const testDSMLModel = "deepseek.v3.2"

func TestStripDSMLLeak(t *testing.T) {
	cases := []struct {
		name, in, want string
	}{
		{"leaked tag after text", "I'll list the files.\n\n<｜DSML｜function_calls", "I'll list the files.\n\n"},
		{"leaked tag with trailing whitespace", "Listing. <｜DSML｜function_calls \n", "Listing. "},
		{"only whitespace and tag", "\n\n<｜DSML｜function_calls", ""},
		{"truncated tag", "Listing.<｜DSML｜function_c", "Listing."},
		{"no tag", "plain text", "plain text"},
		{"full markup kept when upstream did not parse it", "x<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"Bash\">", "x<｜DSML｜function_calls>\n<｜DSML｜invoke name=\"Bash\">"},
		{"tag followed by more text kept", "a<｜DSML｜function_calls then more", "a<｜DSML｜function_calls then more"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, StripDSMLLeak(tc.in))
		})
	}
}

func TestDSMLStreamFilter(t *testing.T) {
	ctx := testBedrockCtx()

	t.Run("tag split across deltas as Bedrock sends it", func(t *testing.T) {
		f := NewDSMLStreamFilter(ctx, testDSMLModel)
		require.NotNil(t, f)
		require.Equal(t, "I'll list", f.Text("I'll list"))
		require.Equal(t, " the files.\n\n", f.Text(" the files.\n\n<｜DSML｜function_c"))
		require.Equal(t, "", f.Text("alls"))
		f.End()
		require.Equal(t, "next block", f.Text("next block"))
	})

	t.Run("markup that continues past the tag is released whole", func(t *testing.T) {
		f := NewDSMLStreamFilter(ctx, testDSMLModel)
		require.Equal(t, "x", f.Text("x<｜DSML｜function_calls"))
		require.Equal(t, "<｜DSML｜function_calls>\n<｜DSML｜invoke", f.Text(">\n<｜DSML｜invoke"))
	})

	t.Run("held whitespace is capped", func(t *testing.T) {
		f := NewDSMLStreamFilter(ctx, testDSMLModel)
		require.Equal(t, "", f.Text("<｜DSML｜function_calls"))
		out := f.Text(strings.Repeat(" ", 64))
		require.Equal(t, "<｜DSML｜function_calls"+strings.Repeat(" ", 64), out)
	})

	t.Run("other models get no filter", func(t *testing.T) {
		f := NewDSMLStreamFilter(ctx, testConverseStreamModel)
		require.Nil(t, f)
		require.Equal(t, "a<｜DSML｜function_calls", f.Text("a<｜DSML｜function_calls"))
		f.End()
	})
}

// converseDSMLResponse is a Converse body shaped like the live deepseek.v3.2 capture.
const converseDSMLResponse = `{"output":{"message":{"role":"assistant","content":[` +
	`{"text":"I'll list the files.\n\n<｜DSML｜function_calls"},` +
	`{"toolUse":{"toolUseId":"tooluse_1","name":"Bash","input":{"command":"ls -la"}}}]}},` +
	`"stopReason":"tool_use","usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15}}`

func converseJSONServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(converseDSMLResponse))
	}))
}

// converseDSMLStreamServer replays the live deepseek.v3.2 ConverseStream event order.
func converseDSMLStreamServer(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		w.WriteHeader(http.StatusOK)
		writeEventStreamEvent(t, w, "messageStart", []byte(`{"role":"assistant"}`))
		writeEventStreamEvent(t, w, "contentBlockDelta", []byte(`{"delta":{"text":"I'll list the files"},"contentBlockIndex":0}`))
		writeEventStreamEvent(t, w, "contentBlockDelta", []byte(`{"delta":{"text":" now.\n\n<｜DSML｜function_c"},"contentBlockIndex":0}`))
		writeEventStreamEvent(t, w, "contentBlockDelta", []byte(`{"delta":{"text":"alls"},"contentBlockIndex":0}`))
		writeEventStreamEvent(t, w, "contentBlockStop", []byte(`{"contentBlockIndex":0}`))
		writeEventStreamEvent(t, w, "contentBlockStart", []byte(`{"start":{"toolUse":{"toolUseId":"tooluse_1","name":"Bash"}},"contentBlockIndex":1}`))
		writeEventStreamEvent(t, w, "contentBlockDelta", []byte(`{"delta":{"toolUse":{"input":"{\"command\": \"ls -la\"}"}},"contentBlockIndex":1}`))
		writeEventStreamEvent(t, w, "contentBlockStop", []byte(`{"contentBlockIndex":1}`))
		writeEventStreamEvent(t, w, "messageStop", []byte(`{"stopReason":"tool_use"}`))
		writeEventStreamEvent(t, w, "metadata", []byte(`{"usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15},"metrics":{"latencyMs":1}}`))
	}))
}

// drainToJSON collects a provider stream and returns every chunk serialized, in order.
func drainToJSON(t *testing.T, stream chan *schemas.BifrostStreamChunk) string {
	t.Helper()
	var b strings.Builder
	timeout := time.After(5 * time.Second)
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				return b.String()
			}
			require.Nil(t, chunk.BifrostError, "unexpected error chunk: %+v", chunk.BifrostError)
			out, err := sonic.Marshal(chunk)
			require.NoError(t, err)
			b.Write(out)
			b.WriteByte('\n')
		case <-timeout:
			t.Fatal("timed out draining stream")
		}
	}
}

func requireDSMLStripped(t *testing.T, out string) {
	t.Helper()
	require.NotContains(t, out, "DSML", "leaked DSML tag reached the client")
	require.Contains(t, out, "I'll list the files", "assistant text must survive")
	require.Contains(t, out, "ls -la", "tool call must survive")
}

func TestDSMLLeakStripped_ConverseChat(t *testing.T) {
	ts := converseJSONServer(t)
	defer ts.Close()
	req := testChatRequest()
	req.Model = testDSMLModel
	resp, bifrostErr := newTestProviderWithServer(t, ts).ChatCompletion(testBedrockCtx(), testBedrockKey(), req)
	require.Nil(t, bifrostErr)
	out, err := sonic.Marshal(resp)
	require.NoError(t, err)
	requireDSMLStripped(t, string(out))
}

func TestDSMLLeakStripped_ConverseResponses(t *testing.T) {
	ts := converseJSONServer(t)
	defer ts.Close()
	req := testResponsesRequest()
	req.Model = testDSMLModel
	resp, bifrostErr := newTestProviderWithServer(t, ts).Responses(testBedrockCtx(), testBedrockKey(), req)
	require.Nil(t, bifrostErr)
	out, err := sonic.Marshal(resp)
	require.NoError(t, err)
	requireDSMLStripped(t, string(out))
}

func TestDSMLLeakStripped_ConverseChatStream(t *testing.T) {
	ts := converseDSMLStreamServer(t)
	defer ts.Close()
	req := testChatRequest()
	req.Model = testDSMLModel
	stream, bifrostErr := newTestProviderWithServer(t, ts).ChatCompletionStream(testBedrockCtx(), noopPostHookRunner, nil, testBedrockKey(), req)
	require.Nil(t, bifrostErr)
	requireDSMLStripped(t, drainToJSON(t, stream))
}

func TestDSMLLeakStripped_ConverseResponsesStream(t *testing.T) {
	ts := converseDSMLStreamServer(t)
	defer ts.Close()
	req := testResponsesRequest()
	req.Model = testDSMLModel
	stream, bifrostErr := newTestProviderWithServer(t, ts).ResponsesStream(testBedrockCtx(), noopPostHookRunner, nil, testBedrockKey(), req)
	require.Nil(t, bifrostErr)
	requireDSMLStripped(t, drainToJSON(t, stream))
}

// Other models pass text through untouched, tag included.
func TestDSMLLeakKept_OtherModelsConverseStream(t *testing.T) {
	ts := converseDSMLStreamServer(t)
	defer ts.Close()
	req := testChatRequest()
	req.Model = testConverseStreamModel
	stream, bifrostErr := newTestProviderWithServer(t, ts).ChatCompletionStream(testBedrockCtx(), noopPostHookRunner, nil, testBedrockKey(), req)
	require.Nil(t, bifrostErr)
	require.Contains(t, drainToJSON(t, stream), "DSML")
}

func BenchmarkDSMLStreamFilter(b *testing.B) {
	ctx := testBedrockCtx()
	text := "I'll list the files in the current directory and then read README.md."
	b.Run("other model (nil filter)", func(b *testing.B) {
		f := NewDSMLStreamFilter(ctx, testConverseStreamModel)
		ev := &BedrockStreamEvent{Delta: &BedrockContentBlockDelta{Text: &text}}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			f.filterConverseEvent(ev)
		}
	})
	b.Run("deepseek text without tag", func(b *testing.B) {
		f := NewDSMLStreamFilter(ctx, testDSMLModel)
		ev := &BedrockStreamEvent{Delta: &BedrockContentBlockDelta{Text: &text}}
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			f.filterConverseEvent(ev)
		}
	})
	b.Run("deepseek leaked tag", func(b *testing.B) {
		f := NewDSMLStreamFilter(ctx, testDSMLModel)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			f.Text(" now.\n\n<｜DSML｜function_c")
			f.Text("alls")
			f.End()
		}
	})
	b.Run("gate check", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = IsDSMLLeakModel(ctx, testConverseStreamModel)
		}
	})
}
